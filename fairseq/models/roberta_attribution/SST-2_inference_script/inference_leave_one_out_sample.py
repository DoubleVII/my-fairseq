from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn
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
samples = [
    "this one is definitely one to skip , even for horror movie fanatics .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious failure .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious .",
]
labels = ["0", "1", "0"]
labels = ["0", "1", "1"]


scores = []
preds = []


def leave_one_out(tokens, probs=None, pred=None):
    assert tokens.dim() == 1
    if probs is None:
        prediction, _ = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_compositions=False,
            dropout_tokens=None,
        )
        probs = nn.functional.softmax(prediction, dim=-1)
    if pred is None:
        pred = probs.argmax(dim=-1).item()
    else:
        assert probs.argmax(dim=-1).item() == pred
    scores = torch.zeros((1, tokens.size(0)))
    for i in range(1, len(tokens)):
        dropout_tokens = torch.LongTensor([[i]]).to(tokens)
        prediction_drop, _ = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_compositions=False,
            dropout_tokens=dropout_tokens,
        )
        probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
        scores[0, i] = probs[0, pred] - probs_drop[0, pred]

    return scores


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
    roberta.float()

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

        probs = nn.functional.softmax(prediction, dim=-1)

        loo_scores = leave_one_out(tokens, probs, pred)
        loo_scores = loo_scores[0]

        words_map, words = bpe_map(sent)

        word_scores = torch.zeros((len(words),)).to(loo_scores)
        for index in range(len(words)):
            word_scores[index] = loo_scores[words_map[index]].sum(dim=0)

        scores.append(word_scores.cpu())
        preds.append(pred)
        # pred_compositions = compositions[
        #     0, 1:, pred
        # ]  # delete 2 element at the front (bias and <bos> composition)

        pred_label = label_fn(pred)

        import pdb

        pdb.set_trace()
        print("pred: ", pred_label, "label: ", label)
