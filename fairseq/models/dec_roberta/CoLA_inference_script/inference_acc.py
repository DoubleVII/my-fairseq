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
    assert test_set in ["dev", "test"]


data_len = 0
with open(f"/home/yangs/data/glue_data/MNLI/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

logger.info(f"infering in {test_set} dataset")


roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/MNLI-bin",
)


drop_values = {}

with torch.no_grad():
    roberta.half()

    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/MNLI/{test_set}.tsv") as fin:
        fin.readline()

        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent0, sent1 = tokens[8], tokens[9]
                label = tokens[-1]
            else:
                sent0, sent1 = tokens[8], tokens[9]
                label = None
            tokens = roberta.encode(sent0, sent1)

            prediction = roberta.predict(
                "sentence_classification_head", tokens, return_logits=True,
            )

            pred = prediction.argmax(dim=-1).item()

            prediction_label = label_map[pred]
            if label is not None:
                ncorrect += int(prediction_label == label)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")

    logger.info(f"Acc: {ncorrect/nsamples}")

    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
