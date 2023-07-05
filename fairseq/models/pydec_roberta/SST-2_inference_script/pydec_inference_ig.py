from fairseq.models.pydec_roberta import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from torch import Tensor
import pydec
import numpy as np

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
    checkpoint_file="checkpoint.pydec.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


def get_top_indices(scores, pred, k):
    scores = scores[:, pred]
    _, top_indices = scores.topk(k=k)
    return top_indices


# def run_dec(model, tokens):
#     top_indices = None
#     pred = None
#     orig_prediction = None
#     num_steps = 20

#     orig_prediction = model.predict(
#         "sentence_classification_head",
#         tokens,
#         return_logits=True,
#     )
#     pred = orig_prediction.argmax(dim=-1).item()

#     scores = []

#     # def mul_alpha(input: pydec.Composition, alpha: Tensor, index: int):
#     #     input()[index] *= alpha.to(input)
#     #     input.init_recovery()
#     #     return input

#     def mul_alpha(input: pydec.Composition, alpha: Tensor, index: int):
#         new_components = input()[index] * alpha.to(input)
#         input()[index] *= 0
#         new_residual = input.c_sum(enforce=True)
#         out = pydec._from_replce(new_components[None], new_residual, enforce=True)
#         out.init_recovery()
#         return out

#     import functools

#     for j in range(tokens.size(-1)):
#         alpha_scores = None
#         for alpha in torch.from_numpy(
#             np.linspace(0, 1.0, num=num_steps, endpoint=False)
#         ):
#             c_prediction = model.predict(
#                 "sentence_classification_head",
#                 tokens,
#                 return_logits=True,
#                 decompose=functools.partial(mul_alpha, alpha=alpha, index=j),
#                 dropout_tokens=top_indices,
#             )
#             if alpha_scores is None:
#                 alpha_scores = c_prediction()[0]
#             else:
#                 alpha_scores += c_prediction()[0]
#         alpha_scores /= num_steps
#         scores.append(alpha_scores)
#     scores = torch.stack(scores, dim=0)
#     scores = scores[1:, 0]  # delete 1 element at the front (<bos> component)
#     top_indices = get_top_indices(scores, pred=pred, k=scores.size(0))
#     top_indices = top_indices + 1
#     return top_indices, pred, orig_prediction


def run_dec(model, tokens):
    top_indices = None
    pred = None
    orig_prediction = None
    num_steps = 20

    orig_prediction = model.predict(
        "sentence_classification_head", tokens, return_logits=True,
    )
    pred = orig_prediction.argmax(dim=-1).item()

    scores = []

    def mul_alpha(input: pydec.Composition, index: int):
        new_components = input()[index].clone()
        input()[index] = 0
        new_residual = input.c_sum(enforce=True)
        out = pydec._from_replce(new_components[None], new_residual, enforce=True)
        out._component_tensor = out._component_tensor.expand(-1, -1, 20, -1).clone()
        out._residual_tensor = out._residual_tensor.expand(-1, 20, -1).clone()
        alphas = torch.from_numpy(
            np.linspace(0, 1.0, num=num_steps, endpoint=False)
        ).to(input)

        out._component_tensor *= alphas.unsqueeze(-1)
        out.init_recovery()
        return out

    import functools

    for j in range(tokens.size(-1)):
        alpha_scores = None
        c_prediction = model.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            decompose=functools.partial(mul_alpha, index=j),
            dropout_tokens=top_indices,
        )
        alpha_scores = c_prediction()[0]
        alpha_scores = alpha_scores.mean(dim=0, keepdim=True)
        scores.append(alpha_scores)
    scores = torch.stack(scores, dim=0)
    scores = scores[1:, 0]  # delete 1 element at the front (<bos> component)
    top_indices = get_top_indices(scores, pred=pred, k=scores.size(0))
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
    with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
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
            # if nsamples != 204:
            #     nsamples += 1
            #     continue
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
                label = tokens[1]
            else:
                sent = tokens[1]
                label = None
            tokens = roberta.encode(sent)
            top_indices, pred, prediction = run_dec(roberta, tokens)

            probs = nn.functional.softmax(prediction, dim=-1)
            AOPC(tokens, top_indices, probs)

            if label is not None:
                ncorrect += int(label_fn(pred) == label)

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
    if ncorrect != 0:
        logger.info(f"acc: {float(ncorrect/nsamples)*100} %")

    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
