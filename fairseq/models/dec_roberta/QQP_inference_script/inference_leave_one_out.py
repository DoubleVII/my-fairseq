from fairseq.models.pydec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
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


def leave_one_out(tokens, probs=None, pred=None):
    assert tokens.dim() == 1
    if probs is None:
        prediction = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            dropout_tokens=None,
        )
        probs = nn.functional.softmax(prediction, dim=-1)
    if pred is None:
        pred = probs.argmax(dim=-1).item()
    else:
        assert probs.argmax(dim=-1).item() == pred
    scores = torch.zeros((1, tokens.size(0)))
    for i in range(1, len(tokens)):
        dropout_tokens = torch.LongTensor([[i]]).to(tokens)
        prediction_drop = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            dropout_tokens=dropout_tokens,
        )
        probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
        scores[0, i] = probs[0, pred] - probs_drop[0, pred]

    return scores


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
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent0, sent1 = tokens[1], tokens[2]
            tokens = roberta.encode(sent0, sent1)
            if len(tokens) > 130:
                logger.info(f"Omited: too much tokens {len(tokens)}")
                nsamples += 1
                continue

            prediction = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                dropout_tokens=None,
            )
            pred = prediction.argmax(dim=-1).item()

            probs = nn.functional.softmax(prediction, dim=-1)

            scores = leave_one_out(tokens, probs, pred)
            scores = scores[0, 1:]

            _, top_indices = torch.topk(scores, k=len(tokens) - 1)
            top_indices = top_indices + 1
            AOPC(tokens, top_indices, probs)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
                # logger.info(f"current AOPC: {torch.mean(torch.stack(drop_values))}")
    if isinstance(drop_values, dict):
        for k, v in drop_values.items():
            logger.info(f"AOPC for k={k}: {torch.mean(torch.stack(drop_values[k]))}")
    else:
        drop_values = torch.stack(drop_values)
        logger.info(f"AOPC: {torch.mean(drop_values)}")
