from fairseq.models.roberta_lrp import RobertaModel
import torch
import torch.nn as nn
import random
import sys
import logging
from AOPC_tools import AOPC

random.seed(10)
torch.manual_seed(10)

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
    checkpoint_file="checkpoint_best.lrp.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


drop_values = {}

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
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent = tokens[1]
            _, sent = tokens[0], tokens[1]
            tokens = roberta.encode(sent)

            prediction = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                record=True,
            )

            pred = prediction.argmax(dim=-1).item()

            relevance = prediction
            relevance[0, 1 - pred] = 0.0

            out_relevance = roberta.relprop(relevance)
            roberta.clear_record()
            out_relevance = out_relevance.sum(-1)[0, 1:]
            _, top_indices = torch.topk(out_relevance, k=len(out_relevance))
            top_indices = top_indices + 1

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
