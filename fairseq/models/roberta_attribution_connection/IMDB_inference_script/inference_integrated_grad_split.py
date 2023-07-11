from datetime import datetime
from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
import torch.nn.functional as F
import os

logger = logging.getLogger(__name__)


def cut_tokens(tokens: torch.Tensor):
    if len(tokens) > 512:
        tokens = tokens[:512]
        tokens[-1] = 2
    return tokens


assert len(sys.argv) > 1
test_set = sys.argv[1]
assert test_set[:-1] == "test_subset"

logger.info(f"infering in {test_set} dataset")

state_dict = None
if os.path.exists(f"{test_set}_dict.pt"):
    state_dict = torch.load(f"{test_set}_dict.pt")

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.attribution.pt",
    data_name_or_path="/home/yangs/data/data-bin/IMDB-bin",
)

# preds = torch.load("checkpoints/dev_preds.pt")

drop_values = [] if state_dict is None else state_dict["drop_values"]
consuming_time = 0 if state_dict is None else state_dict["consuming_time"]

start_num = len(drop_values)
dataset_len = 500

if state_dict is not None:
    logger.info(
        f"loaded state dict, starting from {start_num}, consuming time {consuming_time}s."
    )


def extract_features(model, src_tokens, token_embeddings):
    encoder_out = model.encoder.sentence_encoder(
        src_tokens, token_embeddings=token_embeddings
    )
    # T x B x C -> B x T x C
    features = encoder_out["encoder_out"][0].transpose(0, 1)
    # inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
    return features


def predict(
    model,
    head: str,
    tokens: torch.LongTensor,
    token_embeddings: torch.Tensor,
    return_logits: bool = False,
):
    features = extract_features(model, tokens, token_embeddings,)

    logits, _ = model.classification_heads[head](features, None)
    if return_logits:
        # return logits
        return logits
    return F.log_softmax(logits, dim=-1)


def compute_grad(model, input_tokens: torch.LongTensor, pred):
    import numpy as np

    assert len(input_tokens.size()) == 2
    assert input_tokens.size(0) == 1  # only available when batch = 1

    input_tokens = input_tokens.to(device=model.device)

    embed_tokens = model.model.encoder.sentence_encoder.embed_tokens
    num_steps = 20
    with torch.enable_grad():
        model.requires_grad_(True)
        all_grads = []
        for j in range(input_tokens.size(-1)):
            sum_grads = None
            orig_embed = embed_tokens(input_tokens).clone().detach()[0, j, :]

            alphas = torch.from_numpy(
                np.linspace(0, 1.0, num=num_steps, endpoint=False)
            ).to(orig_embed)
            embed = embed_tokens(input_tokens).clone().detach()  # 1 x T x C
            embed = embed.repeat((len(alphas), 1, 1))  # steps x T x C
            embed[:, j, :] *= alphas[:, None]

            # step0
            embed0 = embed[: len(embed) // 2]
            embed0.requires_grad_(True)
            model.zero_grad()
            lprobs = predict(
                model.model, "sentence_classification_head", input_tokens, embed0
            )
            loss = -torch.sum(lprobs[:, pred])
            loss.backward()

            sum_grads = embed0.grad[:, j, :].clone().detach().sum(dim=0)

            # step1
            embed1 = embed[len(embed) // 2 :]
            embed1.requires_grad_(True)
            model.zero_grad()
            lprobs = predict(
                model.model, "sentence_classification_head", input_tokens, embed1
            )
            loss = -torch.sum(lprobs[:, pred])
            loss.backward()

            sum_grads += embed1.grad[:, j, :].clone().detach().sum(dim=0)
            sum_grads = sum_grads / num_steps
            sum_grads *= orig_embed
            all_grads.append(sum_grads)

        all_grads = torch.stack(all_grads, dim=0)
        all_grads = all_grads.sum(dim=-1)
        all_grads = torch.abs(all_grads) / torch.norm(all_grads, dim=0, p=1)
        return all_grads


with torch.no_grad():
    roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    # ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/imdb_data/IMDB/{test_set}.tsv") as fin:
        fin.readline()
        logger.info("begin inference")
        start_time = datetime.now()

        for index, line in enumerate(fin):
            if index < start_num:
                continue
            tokens = line.strip().split("\t")
            sent, target = tokens[0], tokens[1]
            tokens = roberta.encode(sent)
            tokens = cut_tokens(tokens)
            prediction, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=False,
                dropout_tokens=None,
            )
            pred = prediction.argmax(dim=-1).item()

            all_grads = compute_grad(roberta, tokens[None,], pred)

            all_grads = all_grads[1:]
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(all_grads, k=tokens_to_keep)
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
            start_num += 1
            if start_num % 50 == 0:
                print("process: ", int(start_num / dataset_len * 100), "%")
                saving_time = datetime.now()
                consuming_time += (saving_time - start_time).days * 24 * 3600 + (
                    saving_time - start_time
                ).seconds
                state_dict = {}
                state_dict["drop_values"] = drop_values
                state_dict["consuming_time"] = consuming_time
                torch.save(state_dict, f"{test_set}_dict.pt")

        logger.info("end inference")
    drop_values = torch.stack(drop_values)
    logger.info("AOPC: {}".format(torch.mean(drop_values).item()))
    saving_time = datetime.now()
    consuming_time += (saving_time - start_time).days * 24 * 3600 + (
        saving_time - start_time
    ).seconds
    logger.info(f"consuming time: {consuming_time}s")
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
