from fairseq.models.roberta_ACD import RobertaModel
import torch
import torch.nn as nn

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.acd.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)

preds = torch.load("checkpoints/dev_preds.pt")

drop_values = []


def ACD(tokens, pred=None):
    assert tokens.dim() == 1
    # if probs is None:
    #     prediction, _ = roberta.predict(
    #         "sentence_classification_head",
    #         tokens,
    #         return_logits=True,
    #         return_compositions=False,
    #         dropout_tokens=None,
    #     )
    #     probs = nn.functional.softmax(prediction, dim=-1)
    scores = torch.zeros((1, tokens.size(0)))
    for i in range(1, len(tokens)):
        _, compositions = roberta.predict(
            "sentence_classification_head",
            tokens,
            return_logits=True,
            decomposition_index=i,
            dropout_tokens=None,
        )
        scores[0, i] = compositions[0, 2, pred]

    return scores


with torch.no_grad():
    roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open("/home/yangs/data/glue_data/SST-2/dev.tsv") as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            sent, target = tokens[0], tokens[1]
            tokens = roberta.encode(sent)
            prediction, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                decomposition_index=None,
                dropout_tokens=None,
            )

            scores = ACD(tokens, preds[index])
            # import pdb
            # pdb.set_trace()
            scores = scores[0, 1:]
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(scores, k=tokens_to_keep)
            top_indices = top_indices + 1
            prediction_drop, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                decomposition_index=None,
                dropout_tokens=top_indices,
            )

            probs = nn.functional.softmax(prediction, dim=-1)
            probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            drop_values.append(probs[0, preds[index]] - probs_drop[0, preds[index]])
            nsamples += 1
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / len(preds) * 100), "%")
    drop_values = torch.stack(drop_values)
    print(torch.mean(drop_values))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
