from fairseq.models.dec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from torch import Tensor
import pydec
from typing import Dict, List

logger = logging.getLogger(__name__)


samples = [
    "this one is definitely one to skip , even for horror movie fanatics .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious failure .",
    "every nanosecond of the the new guy reminds you that you could be doing something else far more pleasurable .",
]
labels = ["0", "1", "0"]

samples = [
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious failure .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious _ .",
    "if steven soderbergh 's ` solaris ' is a _ it is a glorious _ .",
]
labels = ["0", "1", "0"]


scores = []
preds = []

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


def bpe_map(sent: str):
    res_map: Dict[int, List[int]] = {}
    words = ["<cls>"] + sent.split() + ["<eos>"]
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


with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    pydec.set_decomposition_func("hybrid_decomposition")
    pydec.set_decomposition_args(threshold=0.0)
    logger.info(f"use bias decomposition func: {pydec.get_decomposition_name()}")
    logger.info("begin inference")
    for label, sent in zip(labels, samples):
        tokens = roberta.encode(sent)
        prediction, composition = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_composition=True,
            dropout_tokens=None,
        )
        words_map, words = bpe_map(sent)
        pred = prediction.argmax(dim=-1).item()

        composition = composition[:, 0]._composition_tensor
        word_composition = torch.zeros((len(words), 2)).to(composition)
        for index in range(len(words)):
            word_composition[index] = composition[words_map[index]].sum(dim=0)

        scores.append(word_composition.cpu())
        preds.append(pred)
        # pred_composition = composition[
        #     0, 1:, pred
        # ]  # delete 2 element at the front (bias and <bos> composition)

        pred_label = label_fn(pred)
        # print("pred: ", pred_label, "label: ", label)

        composition = composition[:, pred]
        _, top_indices = composition.topk(k=2)

        import pdb

        pdb.set_trace()

        prediction_drop, _ = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_composition=False,
            dropout_tokens=top_indices,
        )
        drop_pred = prediction.argmax(dim=-1).item()

        nsamples += 1
    logger.info("end inference")
    torch.save({"scores": scores, "preds": preds}, "sample_word_composition.pt")
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
