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

samples = [
    "this one is definitely one to skip , even for horror movie fanatics .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious failure .",
    "every nanosecond of the the new guy reminds you that you could be doing something else far more pleasurable .",
]
labels = ["0", "1", "0"]


scores = []
preds = []


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


from typing import Dict, List


def bpe_map(sent: str):
    res_map: Dict[int, List[int]] = {}
    words = ["<cls>"] + sent.split() + ["<eos>"]
    bpe_tokens: List[str] = roberta.bpe.encode(sent).split()
    bpe_ptr = 0
    for word_index in range(len(words)):
        res_map[word_index] = []
        word = words[word_index]
        if word_index == 0:
            assert word == "<cls>"
            res_map[word_index].append(0)
            continue
        elif word_index == len(words) - 1:
            assert word == "<eos>"
            res_map[word_index].append(bpe_ptr + 1)
        ptr_tokens = []
        while bpe_ptr < len(bpe_tokens):
            res_map[word_index].append(bpe_ptr + 1)
            ptr_tokens.append(roberta.bpe.decode(bpe_tokens[bpe_ptr]).strip())
            bpe_ptr += 1
            ptr_word = "".join(ptr_tokens)
            assert word.startswith(ptr_word)
            if ptr_word == word:
                break
    return res_map, words


with torch.no_grad():
    # roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    for label, sent in zip(labels, samples):
        tokens = roberta.encode(sent)
        prediction, _ = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_compositions=False,
            dropout_tokens=None,
        )

        pred = prediction.argmax(dim=-1).item()

        all_grads = compute_grad(roberta, tokens[None,], pred,)

        words_map, words = bpe_map(sent)

        word_grads = torch.zeros((len(words),)).to(all_grads)
        for index in range(len(words)):
            word_grads[index] = all_grads[words_map[index]].sum(dim=0)

        scores.append(word_grads.cpu())
        preds.append(pred)
        # pred_compositions = compositions[
        #     0, 1:, pred
        # ]  # delete 2 element at the front (bias and <bos> composition)

        pred_label = label_fn(pred)

        import pdb

        pdb.set_trace()
        print("pred: ", pred_label, "label: ", label)
        nsamples += 1
    logger.info("end inference")
# print("| Accuracy: ", float(ncorrect) / float(nsamples))
