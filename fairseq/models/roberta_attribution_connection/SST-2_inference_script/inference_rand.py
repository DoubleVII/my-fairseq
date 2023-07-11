from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn
import random

random.seed(10)
torch.manual_seed(10)

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)

preds = torch.load("checkpoints/dev_preds.pt")

drop_values = []

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
                return_compositions=False,
                dropout_tokens=None,
            )
            tokens_to_keep = int(len(tokens) * 0.2)
            top_indices = list(range(len(tokens) - 1))
            random.shuffle(top_indices)
            top_indices = torch.LongTensor(top_indices[:tokens_to_keep]).to(tokens)
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
            drop_values.append(probs[0, preds[index]] - probs_drop[0, preds[index]])
            nsamples += 1
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / len(preds) * 100), "%")
    drop_values = torch.stack(drop_values)
    print(torch.mean(drop_values))
    # print("| Accuracy: ", float(ncorrect) / float(nsamples))
