from fairseq.models.roberta_lrp import RobertaModel
import torch
import torch.nn as nn
import random


random.seed(10)
torch.manual_seed(10)

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.lrp.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


drop_values = []

with torch.no_grad():
    roberta.float()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    with open("/home/yangs/data/glue_data/SST-2/test.tsv") as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            _, sent = tokens[0], tokens[1]
            tokens = roberta.encode(sent)

            prediction = roberta.predict(
                "sentence_classification_head", tokens, return_logits=True, record=True,
            )

            pred = prediction.argmax(dim=-1).item()

            relevance = prediction
            relevance[0, 1 - pred] = 0.0

            out_relevance = roberta.relprop(relevance)
            roberta.clear_record()
            out_relevance = out_relevance.sum(-1)[0, 1:]
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(out_relevance, k=tokens_to_keep)
            top_indices = top_indices + 1
            prediction_drop = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                dropout_tokens=top_indices,
            )

            probs = nn.functional.softmax(prediction, dim=-1)
            probs_drop = nn.functional.softmax(prediction_drop, dim=-1)
            drop_values.append(probs[0, pred] - probs_drop[0, pred])
            nsamples += 1
            # print(nsamples)
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / 1821 * 100), "%")
    drop_values = torch.stack(drop_values)
    print(torch.mean(drop_values))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
