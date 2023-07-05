from fairseq.models.pydec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
import pydec
from AOPC_tools import AOPC
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

logger = logging.getLogger(__name__)


def cut_tokens(tokens: torch.Tensor, max_tokens: int = 400):
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        tokens[-1] = 2
    return tokens


test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test", "test_subset"]

data_len = 0
with open(f"/home/yangs/data/imdb_data/IMDB/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

logger.info(f"infering in {test_set} dataset")

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.pydec.pt",
    data_name_or_path="/home/yangs/data/data-bin/IMDB-bin",
)


drop_values = {}


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


with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/imdb_data/IMDB/{test_set}.tsv") as fin:
        fin.readline()
        pydec.core.decOVF.set_decomposition_func("hybrid_decomposition")
        # pydec.set_bias_decomposition_func("norm_decomposition")
        # pydec.set_bias_decomposition_args(p=float("inf"))
        # pydec.set_bias_decomposition_func("abs_decomposition")
        pydec.core.decOVF.set_decomposition_args(threshold=0.1)
        # pydec.set_bias_decomposition_func("sign_decomposition")
        # pydec.set_bias_decomposition_func("hybrid_decomposition_value_threshold")
        # pydec.set_bias_decomposition_func("sign_decomposition_value_threshold")

        logger.info(
            f"use bias decomposition func: {pydec.core.decOVF.get_decomposition_name()}"
        )
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            sent, target = tokens[0], tokens[1]
            tokens = roberta.encode(sent)
            tokens = cut_tokens(tokens)
            top_indices, pred, prediction = run_dec(roberta, tokens)

            probs = nn.functional.softmax(prediction, dim=-1)

            AOPC(roberta, drop_values, tokens, top_indices, probs)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    for k, v in drop_values.items():
        logger.info(f"AOPC for k={k}: {torch.mean(torch.stack(drop_values[k]))}")
    for k, v in drop_values.items():
        print(f"{torch.mean(torch.stack(drop_values[k]))}", end=",")
