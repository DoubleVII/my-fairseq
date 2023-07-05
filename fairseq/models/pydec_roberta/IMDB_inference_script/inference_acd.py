from fairseq.models.roberta_ACD import RobertaModel
import torch
import torch.nn as nn
import sys
import logging

logger = logging.getLogger(__name__)


def cut_tokens(tokens: torch.Tensor):
    if len(tokens) > 256:
        tokens = tokens[:256]
        tokens[-1] = 2
    return tokens


test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test", "test_subset"]

print(f"infering in {test_set} dataset")

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.acd.pt",
    data_name_or_path="/home/yangs/data/data-bin/IMDB-bin",
)

# preds = torch.load("checkpoints/dev_preds.pt")

drop_values = []

dataset_len = 2500

if test_set == "test":
    dataset_len = 25000
else:
    dataset_len = 2000


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
    roberta.double()

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
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(scores, k=tokens_to_keep)
            top_indices = top_indices + 1

            prediction_drop, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                decomposition_index=None,
                dropout_tokens=top_indices,
            )

            probs = nn.functional.softmax(prediction, dim=-1)
            probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            drop_values.append(probs[0, pred] - probs_drop[0, pred])
            nsamples += 1
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / dataset_len * 100), "%")
        logger.info("end inference")
    drop_values = torch.stack(drop_values)
    logger.info("AOPC: {}".format(torch.mean(drop_values).item()))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
