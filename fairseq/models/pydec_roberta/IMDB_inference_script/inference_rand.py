from fairseq.models.dec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
import random

random.seed(10)
torch.manual_seed(10)

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
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/IMDB-bin",
)

# preds = torch.load("checkpoints/dev_preds.pt")

drop_values = []

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
                return_composition=False,
                dropout_tokens=None,
            )
            # import pdb
            # pdb.set_trace()
            pred = prediction.argmax(dim=-1).item()
            tokens_to_keep = int(round(len(tokens) * 0.2))
            top_indices = list(range(len(tokens) - 1))
            random.shuffle(top_indices)
            top_indices = torch.LongTensor(top_indices[:tokens_to_keep]).to(tokens)
            top_indices = top_indices + 1
            prediction_drop, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_composition=False,
                dropout_tokens=top_indices,
            )

            probs = nn.functional.softmax(prediction, dim=-1)
            probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            drop_values.append(probs[0, pred] - probs_drop[0, pred])
            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
                logger.info(
                    "AOPC: {}".format(torch.mean(torch.stack(drop_values)).item())
                )
        logger.info("end inference")
    drop_values = torch.stack(drop_values)
    logger.info("AOPC: {}".format(torch.mean(drop_values).item()))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
