from fairseq.models.pydec_roberta import RobertaModel
import torch
import torch.nn as nn
import logging
import sys
import os
from AOPC_tools import AOPC


from typing import List
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


def cut_tokens(tokens: torch.Tensor, max_tokens: int = 120):
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        tokens[-1] = 2
    return tokens


roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint.dec.pt",
    data_name_or_path="/home/yangs/data/data-bin/RTE-bin",
)

test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test"]
data_len = 0
with open(f"/home/yangs/data/glue_data/RTE/{test_set}.tsv") as fin:
    fin.readline()
    for line in fin:
        data_len += 1

assert test_set == "test"
logger.info(f"infering in {test_set} dataset")

drop_values = {}


explainer = LimeTextExplainer(
    bow=False, mask_string=str(roberta.model.encoder.sentence_encoder.padding_idx)
)


def lime_explain(tokens, pred, num_features):
    tokens = tokens[2:]  # remove [CLS]

    def pred_fn(input: List[str]):
        # import pdb

        # pdb.set_trace()
        input = [
            torch.tensor([0] + [int(token) for token in sent.split()], dtype=torch.long)
            for sent in input
        ]  # [0] is [CLS]
        # max_length = max([len(t) for t in input])
        # batched_input = torch.zeros((len(input), max_length), dtype=torch.long)
        batched_input = torch.stack(input, dim=0)
        # batched_input[:] = roberta.model.encoder.sentence_encoder.padding_idx
        # for i in range(len(input)):
        #     batched_input[i, : len(input[i])] = input[i]

        prediction = roberta.predict(
            "sentence_classification_head",
            batched_input,
            return_logits=True,
            dropout_tokens=None,
        )
        return torch.softmax(prediction, -1).cpu().numpy()

    exp = explainer.explain_instance(
        tokens,
        pred_fn,
        labels=(pred,),
        num_features=num_features,
        num_samples=1024,
    )
    assert pred in exp.as_map()

    top_indices = torch.tensor([t[0] for t in exp.as_map()[pred]], dtype=torch.long)
    return top_indices


with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    # nsamples, drop_values = load()
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/RTE/{test_set}.tsv") as fin:
        fin.readline()
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
            else:
                sent0, sent1 = tokens[1], tokens[2]
            tokens = roberta.encode(sent0, sent1)

            tokens = cut_tokens(tokens)

            prediction = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                dropout_tokens=None,
            )

            pred = prediction.argmax(dim=-1).item()

            probs = nn.functional.softmax(prediction, dim=-1)

            tokens_to_keep = int(round(len(tokens) * (50 / 100)))
            top_indices = lime_explain(
                " ".join([str(token) for token in tokens.tolist()]),
                pred,
                tokens_to_keep,
            )
            top_indices = top_indices + 1
            AOPC(roberta, drop_values, tokens, top_indices, probs)

            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    for k, v in drop_values.items():
        logger.info(f"AOPC for k={k}: {torch.mean(torch.stack(drop_values[k]))}")
    for k, v in drop_values.items():
        print(f"{torch.mean(torch.stack(drop_values[k]))}", end=",")
