from fairseq.models.roberta_attribution import RobertaModel
from fairseq.models.transformer_attribution.attribution_utils import CompositionModual
import torch
import torch.nn as nn
import sys
import logging

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
    checkpoint_file="checkpoint_best.attribution.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


drop_values = []

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
        CompositionModual.set_bias_decomposition_func("hybrid_decomposition")
        logger.info(
            f"use bias decomposition func: {CompositionModual.bias_decomposition_name}"
        )
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent = tokens[1]
            tokens = roberta.encode(sent)
            prediction, compositions = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=True,
                dropout_tokens=None,
            )
            # import pdb
            # pdb.set_trace()
            pred = prediction.argmax(dim=-1).item()
            pred_compositions = compositions[
                0, 2:, pred
            ]  # delete 2 element at the front (bias and <bos> composition)
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(pred_compositions, k=tokens_to_keep)
            top_indices = top_indices + 1
            prediction_drop, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=False,
                dropout_tokens=top_indices,
            )

            probs = nn.functional.softmax(prediction, dim=-1)
            probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            drop_values.append(probs[0, pred] - probs_drop[0, pred])
            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    drop_values = torch.stack(drop_values)
    logger.info(f"AOPC: {torch.mean(drop_values)}")

    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
