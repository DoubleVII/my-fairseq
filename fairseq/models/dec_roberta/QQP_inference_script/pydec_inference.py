from fairseq.models.pydec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from torch import Tensor
import pydec
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logger = logging.getLogger(__name__)

test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test", "test.new"]


data_len = 0
with open(f"/home/yangs/data/glue_data/QQP/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

assert test_set == "test.new"
logger.info(f"infering in {test_set} dataset")


roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/QQP-bin",
)


def get_top_indices(composition: pydec.Composition, pred: int, k: int):
    # logits = composition.c_sum()

    composition = composition[pred]
    _, top_indices = composition.components.topk(k=k)
    return top_indices


def run_dec(model, tokens):
    top_indices = None
    pred = None
    orig_prediction = None

    c_prediction = model.predict(
        "sentence_classification_head",
        tokens,
        return_logits=True,
        decompose=True,
        dropout_tokens=top_indices,
    )
    orig_prediction = c_prediction.c_sum()
    pred = orig_prediction.argmax(dim=-1).item()
    c_prediction = c_prediction()[
        1:, 0
    ]  # delete 1 element at the front (<bos> component)
    top_indices = get_top_indices(c_prediction, pred=pred, k=c_prediction.numc())
    top_indices = top_indices + 1
    return top_indices, pred, orig_prediction


drop_values = {}
ks = [5, 10, 20, 50]


def AOPC(tokens, top_indices, probs):
    for k in ks:
        if k not in drop_values:
            drop_values[k] = []
        tokens_to_keep = int(round(len(tokens) * (k / 100)))
        assert len(top_indices) >= tokens_to_keep
        prediction_drop = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            dropout_tokens=top_indices[:tokens_to_keep],
        )
        if isinstance(prediction_drop, tuple):
            prediction_drop = prediction_drop[0]

        probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
        drop_values[k].append(probs[0, pred] - probs_drop[0, pred])


with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/QQP/{test_set}.tsv") as fin:
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
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent0, sent1 = tokens[1], tokens[2]
            tokens = roberta.encode(sent0, sent1)
            if len(tokens) > 120:
                logger.info(f"Omited: too much tokens {len(tokens)}")
                nsamples += 1
                continue
            top_indices, pred, prediction = run_dec(roberta, tokens)
            assert not torch.any(torch.isnan(prediction))

            probs = nn.functional.softmax(prediction, dim=-1)
            AOPC(tokens, top_indices, probs)

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
