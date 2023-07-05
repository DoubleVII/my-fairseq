# My-Fairseq

This is a repository cloned from [Fairseq](https://github.com/facebookresearch/fairseq).

To read the original fairseq readme file, see [here](/FAIRSEQ_README.md).

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* You also need to install [PyDec](https://github.com/DoubleVII/pydec) latest version.
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/DoubleVII/my-fairseq
cd my-fairseq
pip install --editable ./
```

# Interpreting the RoBERTa model

## Dataset
We use the sentiment classification dataset SST-2, which has binary labels corresponding to positive and negative sentiments. The model is fine-tuned on the training set and then tested on the test set.

## Inference with finetuned RoBERTa
First prepare your model checkpoint. If you don't have one, read [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.glue.md) to finetune a RoBERTa model.

```python
from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='SST-2-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
```