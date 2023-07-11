# from fairseq.models.roberta_attribution import RobertaModel
# from fairseq.models.roberta import RobertaModel
# import torch

# roberta = RobertaModel.from_pretrained(
#     "checkpoints/",
#     checkpoint_file="checkpoint_best.pt.bkp",
#     data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
# )

# preds = []

# with torch.no_grad():
#     # roberta.double()

#     label_fn = lambda label: roberta.task.label_dictionary.string(
#         [label + roberta.task.label_dictionary.nspecial]
#     )
#     ncorrect, nsamples = 0, 0
#     roberta.cuda()
#     roberta.eval()
#     with open("/home/yangs/data/glue_data/SST-2/test.tsv") as fin:
#         fin.readline()
#         for index, line in enumerate(fin):
#             tokens = line.strip().split("\t")
#             # import pdb
#             # pdb.set_trace()
#             # continue
#             _, sent = tokens[0], tokens[1]
#             tokens = roberta.encode(sent)
#             prediction = (
#                 roberta.predict(
#                     "sentence_classification_head", tokens, return_logits=False
#                 )
#                 .argmax()
#                 .item()
#             )
#             preds.append(prediction)

# print(f"Dev sample:{len(preds)}")

# torch.save(preds, "checkpoints/test_preds.pt")


from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn

roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.attribution.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)

preds = torch.load("checkpoints/test_preds.pt")

drop_values = []

with torch.no_grad():
    roberta.double()

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
            prediction, compositions = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=True,
                dropout_tokens=None,
            )
            # import pdb
            # pdb.set_trace()
            pred_compositions = compositions[
                0, 2:, preds[index]
            ]  # delete 2 element at the front (bias and <bos> composition)
            tokens_to_keep = int(round(len(tokens) * 0.2))
            _, top_indices = torch.topk(pred_compositions, k=tokens_to_keep)
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
