from fairseq.models.roberta_explanation import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from torch import Tensor
import pydec
from typing import Dict, List

logger = logging.getLogger(__name__)

test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test"]


data_len = 0
with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

logger.info(f"infering in {test_set} dataset")


roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.explanation.pt",
    # data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
    data_name_or_path="/home/yangs/data/gpt2_data",
)


def get_top_indices(composition: pydec.Composition, pred: int, k: int):
    # logits = composition.c_sum()

    composition = composition[pred]
    _, top_indices = composition.components.topk(k=k)
    return top_indices


# def run_dec(model, tokens):
#     top_indices = None
#     pred = None
#     orig_prediction = None

#     c_prediction = model.predict(
#         "sentence_classification_head",
#         tokens,
#         return_logits=True,
#         decompose=True,
#         dropout_tokens=top_indices,
#     )
#     orig_prediction = c_prediction.c_sum()
#     pred = orig_prediction.argmax(dim=-1).item()
#     c_prediction = c_prediction()[
#         1:, 0
#     ]  # delete 1 element at the front (<bos> component)
#     top_indices = get_top_indices(c_prediction, pred=pred, k=c_prediction.numc())
#     top_indices = top_indices + 1
#     return top_indices, pred, orig_prediction

distributed_activation_dict = {}
token_activation_dict = {}
word_count = {}


def bpe_map(sent: str):
    res_map: Dict[int, List[int]] = {}
    words = ["<cls>"] + sent.split(" ") + ["<eos>"]
    bpe_tokens: List[str] = roberta.bpe.encode(sent).split()
    bpe_ptr = 0
    for word_index in range(len(words)):
        res_map[word_index] = []
        word = words[word_index]

        if word_index == 0:
            assert word == "<cls>"
            res_map[word_index].append(0)
            continue
        elif word_index == len(words) - 1:
            assert word == "<eos>"
            res_map[word_index].append(bpe_ptr + 1)
        ptr_tokens = []
        while bpe_ptr < len(bpe_tokens):
            res_map[word_index].append(bpe_ptr + 1)
            ptr_tokens.append(roberta.bpe.decode(bpe_tokens[bpe_ptr]).strip())
            bpe_ptr += 1
            ptr_word = "".join(ptr_tokens)
            assert word.startswith(ptr_word)
            if ptr_word == word:
                break
    return res_map, words


def record_distribute(activation_composition, words, words_map):
    token_activation_tensor = activation_composition.c_sum()  # 1 x n
    distributed_activation_tensor = activation_composition.components.sum(
        dim=-1
    )  # n x 1
    for word in words:
        if word not in distributed_activation_dict:
            distributed_activation_dict[word] = 0
            token_activation_dict[word] = 0
            word_count[word] = 0
    for i in range(len(words)):
        word = words[i]
        word_count[word] += 1
        token_activation_dict[word] += token_activation_tensor[0, words_map[i]].mean()
        distributed_activation_dict[word] += distributed_activation_tensor[
            words_map[i], 0
        ].mean()


def get_topk_item(activation_dict: Dict, k=10):
    keys = []
    values = []
    for key, value in activation_dict.items():
        keys.append(key)
        values.append(value / word_count[key])

    values = torch.stack(values)
    _, top_idx = values.topk(k=k)
    out = []
    for idx in top_idx:
        out.append((keys[idx], values[idx].item()))
    return out


with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
        fin.readline()
        pydec.core.decOVF.set_decomposition_func("hybrid_decomposition")
        # pydec.set_bias_decomposition_func("norm_decomposition")
        # pydec.set_bias_decomposition_args(p=float("inf"))
        # pydec.set_bias_decomposition_func("abs_decomposition")
        pydec.core.decOVF.set_decomposition_args(threshold=0)
        # pydec.set_bias_decomposition_func("sign_decomposition")
        # pydec.set_bias_decomposition_func("hybrid_decomposition_value_threshold")
        # pydec.set_bias_decomposition_func("sign_decomposition_value_threshold")

        logger.info(
            f"use bias decomposition func: {pydec.core.decOVF.get_decomposition_name()}"
        )
        logger.info("begin inference")
        for index, line in enumerate(fin):
            # if nsamples != 204:
            #     nsamples += 1
            #     continue
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
                label = tokens[1]
            else:
                sent = tokens[1]
                label = None
            tokens = roberta.encode(sent)
            words_map, words = bpe_map(sent)
            prediction = roberta.predict(
                "not working",
                tokens,
                return_logits=True,
                decompose=True,
                target_neuron=(10, 11),
            )
            record_distribute(prediction, words, words_map)

            # probs = nn.functional.softmax(prediction, dim=-1)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    if ncorrect != 0:
        logger.info(f"acc: {float(ncorrect/nsamples)*100} %")
    print(get_topk_item(distributed_activation_dict, k=20))
    print(get_topk_item(token_activation_dict, k=20))

    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
