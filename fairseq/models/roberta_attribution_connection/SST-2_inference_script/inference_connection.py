from posixpath import split
from fairseq.models.roberta_attribution_connection import RobertaModel
import torch
import torch.nn as nn
import sys
import logging
from typing import Dict, List
from fairseq.models.transformer_attribution.attribution_utils import CompositionModual

logger = logging.getLogger(__name__)


CompositionModual.set_bias_decomposition_func("hybrid_decomposition")

logger.info(f"using bias_decomposition_func: {CompositionModual.bias_decomposition_name}")

samples = [
    "this one is definitely one to skip , even for horror movie fanatics .",
    "if steven soderbergh 's ` solaris ' is a failure it is a glorious failure .",
    "every nanosecond of the the new guy reminds you that you could be doing something else far more pleasurable .",
]
labels = ["0", "1", "0"]


scores = []
preds = []

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.attribution_connection.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


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
    roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    logger.info("begin inference")
    for label, sent in zip(labels, samples):
        tokens = roberta.encode(sent)
        prediction, compositions, extra = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            return_compositions=True,
            dropout_tokens=None,
        )
        words_map, words = bpe_map(sent)
        pred = prediction.argmax(dim=-1).item()
        compositions = compositions[0, 1:]
        word_compositions = torch.zeros((len(words), 2)).to(compositions)
        for index in range(len(words)):
            word_compositions[index] = compositions[words_map[index]].sum(dim=0)

        import pdb

        pdb.set_trace()
        scores.append(word_compositions.cpu())
        preds.append(pred)
        # pred_compositions = compositions[
        #     0, 1:, pred
        # ]  # delete 2 element at the front (bias and <bos> composition)

        pred_label = label_fn(pred)
        print("pred: ", pred_label, "label: ", label)

        nsamples += 1
    logger.info("end inference")
    torch.save({"scores": scores, "preds": preds}, "sample_word_compositions.pt")
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
