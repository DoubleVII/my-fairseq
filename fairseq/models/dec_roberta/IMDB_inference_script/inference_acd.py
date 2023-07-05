from fairseq.models.roberta_ACD import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from AOPC_tools import AOPC

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
    checkpoint_file="checkpoint_best.acd.pt",
    data_name_or_path="/home/yangs/data/data-bin/IMDB-bin",
)

drop_values = {}


def ACD(tokens, pred=None):
    assert tokens.dim() == 1
    # if probs is None:
    #     prediction, _ = roberta.predict(
    #         "sentence_classification_head",
    #         tokens,
    #         return_logits=True,
    #         return_compositions=False,
    #         dropout_tokens=None,
    #     )
    #     probs = nn.functional.softmax(prediction, dim=-1)
    scores = torch.zeros((1, tokens.size(0)))
    for i in range(1, len(tokens)):
        _, compositions = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            decomposition_index=i,
            dropout_tokens=None,
        )
        scores[0, i] = compositions[0, 2, pred]

    return scores


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
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            sent, target = tokens[0], tokens[1]
            tokens = roberta.encode(sent)
            tokens = cut_tokens(tokens)

            prediction, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                decomposition_index=None,
                dropout_tokens=None,
            )

            pred = prediction.argmax(dim=-1).item()
            scores = ACD(tokens, pred)
            scores = scores[0, 1:]
            _, top_indices = torch.topk(scores, k=len(scores))
            top_indices = top_indices + 1
            probs = nn.functional.softmax(prediction, dim=-1)

            AOPC(roberta, drop_values, tokens, top_indices, probs)
            nsamples += 1
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / data_len * 100), "%")
        logger.info("end inference")
    for k, v in drop_values.items():
        logger.info(f"AOPC for k={k}: {torch.mean(torch.stack(drop_values[k]))}")
    for k, v in drop_values.items():
        print(f"{torch.mean(torch.stack(drop_values[k]))}", end=",")
