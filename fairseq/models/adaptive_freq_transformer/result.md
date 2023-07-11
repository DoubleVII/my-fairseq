## dataset setting
|dataset| samples| avg len|
|----|----|----|
|wmt16 test|3003|26.6|
|cat sentences test|1502|53.1|
|split sentences test|3269|24.4|

## len robust transformer

|Model  | BLEU|
|----|----|
|Transfomer(base)               |27.19    |
|&emsp;+rel pos                 |26.9     |
|Transfomer(base,rel pos only)  |27.19    |
|len robust transformer(50%)    |27.26    |
|&emsp;+pos alignment           |27.08    |
|&emsp;&emsp;+rel pos           |27.49    |
|len robust transformer(80%)    |27.52    |
|len robust transformer(90%)    |27.05    |

## cat dataset

|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |10.95    |0.545|
|&emsp;+rel pos                 |-        ||
|Transfomer(base,rel pos only)  |16.64    |0.687|
|len robust transformer(50%)    |18.61    |0.744|
|&emsp;+pos alignment           |16.04    |0.671|
|&emsp;&emsp;+rel pos           |-    ||
|len robust transformer(80%)    |17.76    |0.720|
|len robust transformer(90%)    |17.88    |0.720|

## split dataset

|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |23.41    |-|
|&emsp;+rel pos                 |-        |-|
|Transfomer(base,rel pos only)  |26.79    |-|
|len robust transformer(50%)    |26.86    |-|
|&emsp;+pos alignment           |26.54    |-|
|&emsp;&emsp;+rel pos           |-    |-|
|len robust transformer(80%)    |27.08    |-|
|len robust transformer(90%)    |26.67    |-|


## newstest 2016
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |33.55    |0.996|
|len robust transformer(50%)    |34.35    |0.990|
|len robust transformer(80%)    |34.43    |0.997|

## newstest 2015
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |29.30    |1.012|
|len robust transformer(50%)    |29.92    |1.009|
|len robust transformer(80%)    |29.73    |1.012|

## newstest 2013
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |29.30    |1.012|
|len robust transformer(50%)    |29.92    |1.009|
|len robust transformer(80%)    |29.73    |1.012|

## newstest 2012
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |29.30    |1.012|
|len robust transformer(50%)    |29.92    |1.009|

## iwlst test
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |13.64    |0.997|
|len robust transformer(50%)    |13.95    |0.999|

## iwlst test
|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |12.89    |0.989|
|len robust transformer(50%)    |12.93    |0.989|

## split dataset head

|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |27.53    |1.049|
|len robust transformer(50%)    |27.82    |1.058|
|len robust transformer(80%)    |27.40    |1.047|


## split dataset tail

|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |22.15    |0.956|
|len robust transformer(50%)    |22.30    |0.976|
|len robust transformer(80%)    |21.63    |0.986|

## split dataset rest

|Model  | BLEU| len ratio|
|----|----|----|
|Transfomer(base)               |26.89    |1.017|
|len robust transformer(50%)    |27.00    |1.018|
|len robust transformer(80%)    |27.34    |1.018|

<!-- sen num:  3003
en avg:  26.572760572760572
de avg:  28.075258075258077 -->

<!-- sen num:  1502
en avg:  53.12782956058589
de avg:  56.13182423435419 -->

<!-- sen num:  3269
en avg:  24.410523095747934
de avg:  25.79076170082594 -->


split head
sen num:  266
en avg:  27.383458646616543
de avg:  27.413533834586467
split tail
sen num:  266
en avg:  15.88345864661654
de avg:  18.2406015037594
split rest
sen num:  2737
en avg:  24.950310559006212
de avg:  26.36682499086591


<!-- fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_de_en_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20 -->

<!-- TEXT=nist_zh-en_1.34m
fairseq-preprocess \
    --source-lang zh --target-lang en \
    --trainpref $TEXT/train.bpe \
    --validpref $TEXT/validset.bpe \
    --destdir data-bin/nist_zh_en \
    --workers 20 -->