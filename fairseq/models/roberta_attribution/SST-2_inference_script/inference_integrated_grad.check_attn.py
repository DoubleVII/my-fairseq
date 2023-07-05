from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

logger = logging.getLogger(__name__)

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.attribution.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)

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


drop_values = []

q_grad_share_list = []


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
    features = extract_features(
        model,
        tokens,
        token_embeddings,
    )

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
            embed.requires_grad_(True)
            model.zero_grad()
            lprobs = predict(
                model.model, "sentence_classification_head", input_tokens, embed
            )
            loss = -torch.sum(lprobs[:, pred])
            loss.backward()
            sum_grads = embed.grad[:, j, :].clone().detach().sum(dim=0)
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
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
        fin.readline()
        logger.info("begin inference")

        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            sent, target = tokens[0], tokens[1]
            tokens = roberta.encode(sent)
            prediction, _, attn_states = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=False,
                dropout_tokens=None,
                need_attn=True,
            )
            pred = prediction.argmax(dim=-1).item()

            with torch.enable_grad():
                roberta.requires_grad_(True)
                roberta.zero_grad()
                prediction, _, attn_states = roberta.predict(
                    "sentence_classification_head",
                    tokens,
                    return_logits=True,
                    return_compositions=False,
                    dropout_tokens=None,
                    need_attn=True,
                )
                loss = -torch.sum(prediction[:, pred])
                loss.backward()

                q_grad_share = 0
                for i in range(len(attn_states)):
                    q_grad_share += (
                        attn_states[i][0].grad.norm(dim=-1).mean()
                        / attn_states[i][1].grad.norm(dim=-1).mean()
                    )
                q_grad_share /= len(attn_states)
                q_grad_share_list.append(q_grad_share)

            # all_grads = compute_grad(
            #     roberta,
            #     tokens[
            #         None,
            #     ],
            #     pred,
            # )

            # all_grads = all_grads[1:]
            # tokens_to_keep = int(round(len(tokens) * 0.2))
            # _, top_indices = torch.topk(all_grads, k=tokens_to_keep)
            # top_indices = top_indices + 1

            # prediction_drop, _ = roberta.predict(
            #     "sentence_classification_head",
            #     tokens,
            #     return_logits=True,
            #     return_compositions=False,
            #     dropout_tokens=top_indices,
            # )

            # probs = nn.functional.softmax(prediction, dim=-1)
            # probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            # drop_values.append(probs[0, pred] - probs_drop[0, pred])
            nsamples += 1
            if nsamples % 50 == 0:
                logger.info(f"process: {int(nsamples / data_len * 100)} %")
        logger.info("end inference")
    q_grad_share_avg = sum(q_grad_share_list) / len(q_grad_share_list)
    logger.info(f"avg q grad share: {q_grad_share_avg}")
    # drop_values = torch.stack(drop_values)
    # print(torch.mean(drop_values))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
