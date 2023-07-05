from fairseq.models.dec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from torch import Tensor
import pydec

logger = logging.getLogger(__name__)

iter_num = 1

test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test"]

if len(sys.argv) > 2:
    iter_num = int(sys.argv[2])

data_len = 0
with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

logger.info(f"infering in {test_set} dataset")


logger.info(f"iter: {iter_num}")

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


def get_top_indices(composition: pydec.Composition, pred: int, k: int):
    # logits = composition.c_sum()

    composition = composition[:, pred]
    _, top_indices = composition._composition_tensor.topk(k=k)
    return top_indices


def run_dec(model, tokens):
    tokens_to_keep = int(round(len(tokens) * 0.2))
    mask_nums = iter_num * [tokens_to_keep // iter_num]
    mask_nums[0] = mask_nums[0] + tokens_to_keep - sum(mask_nums)
    while 0 in mask_nums:
        mask_nums.remove(0)
    if mask_nums[0] < 2 and len(mask_nums) > 1:
        mask_nums[0] = mask_nums[0] + 1
        mask_nums[-1] = mask_nums[-1] - 1
        if mask_nums[-1] == 0:
            mask_nums.remove(0)

    top_indices = None
    pred = None
    orig_prediction = None

    for i in range(len(mask_nums)):
        assert mask_nums[i] != 0
        prediction, composition = model.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_composition=True,
            dropout_tokens=top_indices,
        )
        if pred is None:
            pred = prediction.argmax(dim=-1).item()
            orig_prediction = prediction
        pred_composition = composition[
            1:, 0
        ]  # delete 1 element at the front (<bos> component)
        i_top_indices = get_top_indices(pred_composition, pred=pred, k=mask_nums[i])
        i_top_indices = i_top_indices + 1
        if top_indices is None:
            top_indices = i_top_indices
        else:
            top_indices = torch.cat([top_indices, i_top_indices])
    return top_indices, pred, orig_prediction


drop_values = {}
ks = [5, 10, 20, 50]


def AOPC(top_indices, probs):
    for k in ks:
        if k not in drop_values:
            drop_values[k] = []
        prediction_drop, _ = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_compositions=False,
            dropout_tokens=top_indices,
        )

        probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
        drop_values[k].append(probs[0, pred] - probs_drop[0, pred])


with torch.no_grad():
    roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
        fin.readline()
        pydec.set_bias_decomposition_func("none")

        # pydec.set_bias_decomposition_func("hybrid_decomposition")
        # pydec.set_bias_decomposition_func("norm_decomposition")
        # pydec.set_bias_decomposition_args(p=float("inf"))
        # pydec.set_bias_decomposition_func("abs_decomposition")
        # pydec.set_bias_decomposition_args(threshold=0.15)
        # pydec.set_bias_decomposition_func("sign_decomposition")
        # pydec.set_bias_decomposition_func("hybrid_decomposition_value_threshold")
        # pydec.set_bias_decomposition_func("sign_decomposition_value_threshold")

        logger.info(
            f"use bias decomposition func: {pydec.get_bias_decomposition_name()}"
        )
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent = tokens[1]
            tokens = roberta.encode(sent)
            top_indices, pred, prediction = run_dec(roberta, tokens)
            probs = nn.functional.softmax(prediction, dim=-1)
            AOPC(top_indices, probs)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    if isinstance(drop_values, dict):
        for k, v in drop_values.items():
            logger.info(f"AOPC for k={k}: {torch.mean(torch.stack(drop_values[k]))}")
    else:
        drop_values = torch.stack(drop_values)
        logger.info(f"AOPC: {torch.mean(drop_values)}")

    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
