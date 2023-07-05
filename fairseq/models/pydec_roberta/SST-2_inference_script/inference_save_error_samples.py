from fairseq.models.roberta_attribution import RobertaModel
import torch
import torch.nn as nn
import sys
import logging

logger = logging.getLogger(__name__)

test_set = "dev"
if len(sys.argv) > 1:
    test_set = sys.argv[1]
    assert test_set in ["dev", "test"]

assert test_set == "dev", "only 'dev' dataset have label, please use 'dev' dataset"
data_len = 0
with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin:
    fin.readline()
    logger.info("begin inference")
    for line in fin:
        data_len += 1

logger.info(f"infering in {test_set} dataset")


roberta = RobertaModel.from_pretrained(
    "checkpoints/",
    checkpoint_file="checkpoint_best.attribution.pt",
    data_name_or_path="/home/yangs/data/data-bin/SST-2-bin",
)


drop_values = []

with torch.no_grad():
    roberta.double()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    n_error = 0
    roberta.cuda()
    roberta.eval()
    with open(f"/home/yangs/data/glue_data/SST-2/{test_set}.tsv") as fin, open(
        "error_samples.dev.txt", "w"
    ) as save_file:
        fin.readline()
        logger.info("begin inference")
        for index, line in enumerate(fin):
            tokens = line.strip().split("\t")
            if test_set == "dev":
                sent = tokens[0]
                label = tokens[1]
            else:
                sent = tokens[1]
                label = None
            tokens = roberta.encode(sent)
            prediction, _ = roberta.predict(
                "sentence_classification_head",
                tokens,
                return_logits=True,
                return_compositions=False,
                dropout_tokens=None,
            )
            pred = prediction.argmax(dim=-1).item()
            pred_label = label_fn(pred)
            if pred_label != label:
                input_line = line.strip().split("\t")[0]
                save_file.write(f"{label} {input_line}\n")
                n_error += 1
            else:
                ncorrect += 1

            nsamples += 1
            if nsamples % 50 == 0:
                print("process: ", int(nsamples / data_len * 100), "%")
        logger.info("end inference")
    print("| Accuracy: ", float(ncorrect) / float(nsamples))
    print(f"save {n_error} error samples")
