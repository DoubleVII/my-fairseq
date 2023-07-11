
<!-- # iwslt14 BLEU Scores
|  Model   | BLEU  |
|  ----  | ----  |
| tansformer(base)  | 33.04 |
| +layer attn  | 32.61 |
| +layer attn +encoder recurrent| 32.44 |
| +layer attn +max reduction|32.57|

|  Model   | BLEU  |
|  ----  | ----  |
| tansformer(large)  | 34.32 |
| +layer attn  | 34.28 |
| +layer attn +encoder recurrent| 34.79 |

|  Model   | BLEU  |
|  ----  | ----  |
| tansformer(layer=2)  | 30.76 |
| +layer attn  | 26.03 |
| +layer attn +encoder recurrent | ~25 |
| -layer attn | 29.29 |


--- -->
# iwslt14 BLEU Scores


## Monolayer Recurrent Experiment
|Model                      |BLEU       |Param  |
|----                       |----       |----   |
|tansformer(layer=6)        |34.65      |13.7M  |
|monolayer(rec6)            |33.86      |4.4M   |
|monolayer(rec9)            |33.64      |4.4M   |


## Shallow Transformer Experiment

|Model                      |BLEU       |Param  |
|----                       |----       |----   |
|tansformer(layer=2)        |31.30      |6.2M   |
|-layer attn                |30.65      |6.2M   |
|+layer attn                |30.47      |6.8M   |
|+layer attn +encoder rec   |**32.41**  |6.8M   |
|+layer attn +encoder rec\* |31.67      |6.8M   |

\*: only output last recurrent layer

## Base Transofrmer Experiment
|Model                      |BLEU       |Param  |
|----                       |----       |----   |
|tansformer(small)          |34.65      |13.7M  |
|+layer attn                |34.48      |15.2M  |
|+layer attn +encoder rec   |**35.15**  |15.2M  |
*tansformer(small):layer=6, embedding/hidden size=256, ffn hidden=1024, attention head=8*
___
## Recurrent Experiment
|Model                      |BLEU       |Param  |
|----                       |----       |----   |
|recurrent=2                |34.99      |15.2M  |
|recurrent=3                |35.15      |15.2M  |
|recurrent=4                |34.94      |49.3M  |
|recurrent=8                |N/A        |49.3M  |

<br/>


# iwslt14 BLEU Scores(route version)

## Baseline Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(shallow) dropout=0.3|33.42      |8.1M   |
|tansformer(shallow) dropout=0.1|32.77      |8.1M   |
|tansformer(base) dropout=0.3   |34.65      |13.7M  |
|tansformer(base) dropout=0.1   |32.23      |13.7M  |

<!-- ## Baseline Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|layer=3 dropout=0.3|33.42      |8.1M   |
|layer=3 dropout=0.1|32.77      |8.1M   |
|layer=6 dropout=0.3   |34.65      |13.7M  |
|layer=6 dropout=0.1   |32.23      |13.7M  | -->


|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=3)            |33.42      |8.1M   |
|&emsp;+layer linear            |32.14      |8.1M   |
|&emsp;+layer attn              |33.12      |8.9M   |
|&emsp;&emsp;+encoder rec             |34.13      |8.9M   |
|&emsp;&emsp;+encoder rec(D)            |34.64      |8.9M   |
|&emsp;+layer attn(OL)          |33.44      |8.4M   |
|&emsp;+layer attn(T)           |33.50      |8.9M   |
|&emsp;+layer attn(T,OL)        |33.52      |8.4M   |
|&emsp;&emsp;+encoder rec(S)           |34.36      |8.4M   |
|&emsp;&emsp;+encoder rec(D)|34.55      |8.4M   |
|&emsp;&emsp;+encoder rec(S) +decoder rec(D)|33.96      |8.4M   |
|&emsp;&emsp;+rec(D)            |34.73      |8.4M   |

|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=3)            |33.42      |8.1M   |
|&emsp;+encoder rec(D)                |34.86      |8.1M   |
|&emsp;+decoder rec(D)                |34.54      |8.1M   |
|&emsp;+encoder rec(L)                |34.88      |8.1M   |
|&emsp;+decoder rec(L)                |34.63      |8.1M   |
|&emsp;+encoder rec(FGL)              |**35.12**  |8.1M   |
|&emsp;+decoder rec(FGL)              |34.71      |8.1M   |
|&emsp;+encoder rec(FGL*)             |**35.15**  |8.1M   |
|&emsp;+rec(D)                        |**35.14**  |8.1M   |
|&emsp;+rec(FGL)                      |**35.17**  |8.1M   |
|&emsp;+rec(FGL*)                     |**35.54**  |8.1M   |

|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=6)            |34.65      |13.7M  |
|tansformer(layer=3)            |33.42      |8.1M   |
|&emsp;+rec(D)                  |**35.14**  |8.1M   |
|&emsp;&emsp;+layer attn(T,OL)  |34.73      |8.4M   |
|&emsp;+rec(FGL)                |**35.17**  |8.1M   |
|&emsp;+rec(FGL*)               |**35.54**  |8.1M   |
|&emsp;+layer attn(T,OL) +encoder rec(S) +decoder rec(D)|33.96      |8.4M   |


S: shallow connection
D: deep connection
T: time-wise
OL: apply layer attention to output layer of the decoder
L: linear joint connection
FGL: fine-grained linear joint connection
*: similar version but less parameters


## Shallow Transformer Experiment(for quick train speed)

|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=3)            |33.42      |8.1M   |
|+layer linear                  |32.14      |8.1M   |
|+layer attn                    |33.12      |8.9M   |
|+layer attn(OL)                |33.44      |8.4M   |
|+layer attn(T)                 |33.50      |8.9M   |
|+layer attn(T,OL)              |33.52      |8.4M   |
|+encoder rec*                  |34.86      |8.1M   |
|+decoder rec*                  |34.54      |8.1M   |
|+encoder rec#                  |34.88      |8.1M   |
|+decoder rec#                  |34.63      |8.1M   |
|+encoder rec#(T)               |**35.12**  |8.1M   |
|+decoder rec#(T)               |34.71      |8.1M   |
|+encoder rec#(T,I)             |**35.15**  |8.1M   |
|+encoder rec* +decoder rec*    |**35.14**  |8.1M   |
|+encoder rec#(T) +decoder rec#(T)|**35.17**  |8.1M   |
|+encoder rec#(T,I) +decoder rec#(T,I)|**35.54**  |8.1M   |
|+layer attn  +encoder rec      |34.13      |8.9M   |
|+layer attn  +encoder rec*     |34.64      |8.9M   |
|+layer attn(T,OL) +encoder rec |34.36      |8.4M   |
|+layer attn(T,OL) +encoder rec*|34.55      |8.4M   |
|+layer attn(T,OL) +encoder rec +decoder rec*   |33.96      |8.4M   |
|+layer attn(T,OL) +encoder rec* +decoder rec*  |34.73      |8.4M   |
|layer(5,1) +rec(T,I)           |34.74      |7.6M    |


**: deep connection*

*#: linear connection*

*T: time-wise*

*I: an indepentend parameter pattern*

*OL: only use layer attention in last decoder layer*

*The recurrent time = 3*

*"+layer attn  +encoder rec\*" setup use fp32 in training instead of fp16 which will cause gradient error.*

## Transformer(Samll) Experiment
|Model                              |BLEU       |Param  |
|----                               |----       |----   |
|tansformer(layer=6)                |34.65      |13.7M  |
|+encoder rec#(T)                   |**35.20**  |13.7M   |
|+encoder rec#(T) +decoder rec#(T)  |34.26      |13.7M   |
|+encoder rec#(T) +layer attn(T,OL) |34.56      |13.9M   |

___

# wmt14 BLEU Scores(route version)

## Baseline Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(shallow) dropout=0.3|24.54      |14.8M  |
|tansformer(shallow) dropout=0.1|27.89      |14.8M  |
|tansformer(base) dropout=0.3   |30.92      |62.7M  |
|tansformer(base) dropout=0.1   |31.73      |62.7M  |

<!-- ## Baseline Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|layer=3 dropout=0.3|24.54      |14.8M  |
|layer=3 dropout=0.1|27.89      |14.8M  |
|layer=6 dropout=0.3   |30.92      |62.7M  |
|layer=6 dropout=0.1   |31.73      |62.7M  | -->

|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=3)            |29.54      |40.6M  |
|&emsp;+rec(FGL*)              |31.04      |40.6M  |
|&emsp;+rec4(FGL*)             |31.77      |40.6M  |
|tansformer(base)               |31.73      |62.7M  |
|&emsp;+encoder rec(FGL)       |**32.09**      |62.7M  |
|&emsp;+encoder rec4(FGL)      |**32.10**      |62.7M  |

## Transformer(shallow) Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(layer=3)            |29.54      |40.6M  |
|+rec#(T,I)                     |31.04      |40.6M  |
|+rec4#(T,I)                    |31.77      |40.6M  |

## Transformer(base) Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(base)               |31.73      |62.7M  |
|tansformer(large)              |32.73      |213.5M |
|+encoder rec#                  |32.03      |62.7M  |
|+encoder rec#(T)               |32.09      |62.7M  |
|+encoder rec4#(T)              |32.10      |62.7M  |
|EL=6 DL=3  +encoder rec#(T) +decoder rec#(T)|31.73      |N/A  |
|EL=10 DL=3 +encoder rec#(T) +decoder rec#(T)|32.01      |62.7M  |

*EL: encoder layer*

*DL: decoder layer*

## en=>de Experiment
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(base)               |26.53      |62.7M  |
|DLDC(base)                     |26.46      |62.7M  |
|EL=10 DL=3 +encoder rec#(T) +decoder rec#(T)|26.76      |62.7M  |
|EL=10 DL=3 +encoder rec#(T,I) +decoder rec#(T,I)|26.98  |62.7M  |


|Model                          |=>en       |=>de       |Param  |
|----                           |----       |----       |----   |
|tansformer(base)               |31.73      |26.53      |62.7M  |
|&emsp;+encoder rec(D)          |32.03      |-          |62.7M  |
|&emsp;+encoder rec(FGL)        |**32.09**  |-          |62.7M  |
|&emsp;+encoder rec4(FGL)       |**32.10**  |-          |62.7M  |
|DLCL(base)                     |-          |26.46      |62.7M  |
|DLCL-deep(base,25L)            |-          |**28.32**  |122.6M |
|layer=(6,3) +rec(FGL)          |31.73      |-          |-      |
|layer=(10,3) +rec(FGL)         |32.01      |27.05      |62.7M  |
|layer=(10,3) +rec(FGL*)        |-          |27.48      |62.7M  |
|layer=(10,3) +rec(FGL*) +layer linear|-    |26.93      |62.7M  |
|layer=(10,3) +rec(DLCL)        |-          |26.98      |62.7M  |

simple version
|Model                          |BLEU       |Param  |
|----                           |----       |----   |
|tansformer(base)               |26.53      |62.7M  |
|DLCL(base)                     |26.46      |62.7M  |
|DLCL-deep(base,25L)            |**28.32**  |122.6M |
|layer=(10,3) +rec(FGL*)        |27.48      |62.7M  |

# Speed Experiment

<!-- ## iwslt14 Dataset
|Model                          |Param  |Training Speed  |Inference Speed  | 
|----                           |----   |----   |----   |
|tansformer(layer=6)            |13.7M  |198.4 232.1s - 215.25 - 4.645760743321719|47.0 41.2 48.9 43.4s - 45.125 - 2.21606648199446 |
|tansformer(layer=2,recurrent=3)|6.3M   |175.3 168.4s - 171.85 - 5.819028222286878|65.5 69.8 68.0 60.5s - 65.95 - 1.5163002274450341|
|tansformer(layer=9)            |19.2M  |262.9 269.7s - 266.3 - 3.7551633496057075|67.1 61.4 66.7 64.1s - 64.825 - 1.5426147319706902|
|tansformer(layer=3,recurrent=3)|8.1M   |223.0 224.8s - 223.9 - 4.466279589102277|104.7 99.6 100.2 100.4s - 0.9878982464806125|
|tansformer(layer=3,recurrent=3)|8.1M   |NA NAs - NA - NA|39.7 34.9 33.8 35.5 -36.0 - 2.7777777777777777| -->


<!-- ## iwslt14 Dataset
|Model                          |Param  |Training Speed  |Inference Speed  | 
|----                           |----   |----   |----   |
|tansformer(layer=6)            |13.7M  |198.4 232.1s - 215.25 - 4.645760743321719|28.2 28.2- 21606648199446 |
|tansformer(layer=2,recurrent=3)|6.3M   |175.3 168.4s - 171.85 - 5.819028222286878|91.8 93.3 - 1.5163002274450341|
|tansformer(layer=9)            |19.2M  |262.9 269.7s - 266.3 - 3.7551633496057075| - 1.5426147319706902|
|tansformer(layer=3,recurrent=3)|8.1M   |223.0 224.8s - 223.9 - 4.466279589102277| - 0.9878982464806125|
|tansformer(layer=3,recurrent=3)|8.1M   |NA NAs - NA - NA|24.9 24.9 - 2.7777777777777777| -->



## iwslt14 Dataset
|Model                          |Param  |Training   |Inference(tokens/s)  |BLEU|
|----                           |----   |----       |----       |----|
|tansformer(layer=6)            |13.7M  |x1.24      |3455       |34.65|
|tansformer(layer=2,recurrent=3)|6.3M   |x1.55      |3416       |34.92|
|tansformer(layer=9)            |19.2M  |x1.00      |2835       |35.20|
|tansformer(layer=3,recurrent=3)|8.1M   |x1.19      |2722       |35.18|
|layer(5,1) +rec(FGL*)          |7.6M   |-          |3211       |34.74|

<!-- ## wmt14 Dataset
|Model                          |Param  |Training  |Inference  | 
|----                           |----   |----   |----   |
|tansformer(layer=6)            |62.7M  | 1209.3s - 0.8269246671628215|31.9 23.6 29.1 27.5 - 28.025 - 3.568242640499554|
|tansformer(layer=2,recurrent=3)|33.3M  | 1093.0s - 0.9149130832570905|41.0 29.3 35.1 28.9 - 33.575 - 2.978406552494415|
|tansformer(layer=9)            |84.8M  | 1563.8s - 0.639467962655071|31.8 29.2 29.5 31.2 - 30.425 - 3.286770747740345|
|tansformer(layer=3,recurrent=3)|45.0M  | 1378.0s - 0.7256894049346879|56.8 51.5 58.4 56.9 - 55.9 - 1.7889087656529516| -->

## wmt14 Dataset
|Model                          |Param  |Training   |Inference  |
|----                           |----   |----   |----   |
|tansformer(layer=6)            |62.7M  |x1.29  |~~x1.09~~  |
|tansformer(layer=2,recurrent=3)|33.3M  |x1.43  |~~x0.91~~  |
|tansformer(layer=9)            |84.8M  |x1.00  |~~x1.00~~  |
|tansformer(layer=3,recurrent=3)|45.0M  |x1.13  |~~x0.54~~  |



# Length Generalization Experiment

|Model                          |Short  |Long   |Inference  |
|----                           |----   |----   |----   |
|            |62.7M  |x1.29  |~~x1.09~~  |
|tansformer(layer=3)            |33.3M  |x1.43  |~~x0.91~~  |
|&emsp;+rec(D)                  |84.8M  |x1.00  |~~x1.00~~  |
|&emsp;+rec(FGL*)               |45.0M  |x1.13  |~~x0.54~~  |

<table>
   <tr>
      <td rowspan="2">Model</td>
      <td colspan="2">Short</td>
      <td colspan="2">Long</td>
   </tr>
   <tr>
      <td >L<sub>avg</sub></td>
      <td >N-MSE</td>
      <td >L<sub>avg</sub></td>
      <td>N-MSE</td>
   </tr>
   <tr>
      <td>reference</td>
      <td>4.21</td>
      <td>-</td>
      <td>63.92</td>
      <td>-</td>
   </tr>
   <tr>
      <td>tansformer(layer=6)</td>
      <td>5.08</td>
      <td>0.52</td>
      <td>53.62</td>
      <td>4.05</td>
   </tr>
   <tr>
      <td>tansformer(layer=3)</td>
      <td>4.98</td>
      <td>0.46</td>
      <td>51.77</td>
      <td>5.27</td>
   </tr>
   <tr>
      <td>&emsp;+rec(D)</td>
      <td>5.20</td>
      <td>0.62</td>
      <td>55.60</td>
      <td>3.56</td>
   </tr>
   <tr>
      <td>&emsp;+rec(FGL*)</td>
      <td>5.10</td>
      <td>0.50</td>
      <td>55.87</td>
      <td>2.93</td>
   </tr>
</table>

<table>
   <tr>
      <td colspan="2"></td>
      <td >Coder-Wise</td>
      <td >Layer-Wise</td>
      <td >Position-Wise</td>
   </tr>
   <tr>
      <td colspan="2">Encoder/Decoder</td>
      <td >Transformer</td>
      <td >DLCL</td>
      <td>FGL*</td>
   </tr>
   <tr>
      <td rowspan="2">Encoder-Decoder</td>
      <td>Encoder</td>
      <td>Transformer</td>
      <td>DLCL/Layer Linear</td>
      <td>Layer Attn</td>
   </tr>
   <tr>
      <td>Deocder</td>
      <td>Transformer/DLCL</td>
      <td>Layer Linear</td>
      <td>Layer Attn</td>
   </tr>
</table>



# Fitting Experiment



## Recurrent Experiment
|Model                     |BLEU       |Loss   |Param  |
|----                      |----       |----   |----   |
|tansformer(layer=6)       |34.65      |-      |13.7M  |
|student tansformer(layer=6)|34.24     |0.014  |13.7M  |
|monolayer                 |31.98      |0.053  |9.7M   |
|monolayer(rec2)           |32.94      |0.042  |9.7M   |
|monolayer(rec3)           |33.27      |0.039  |9.7M   |
|monolayer(rec4)           |33.41      |0.037  |9.7M   |
|monolayer(rec5)           |33.32      |0.037  |9.7M   |
|monolayer(rec6)           |33.42      |0.037  |9.7M   |
|monolayer(rec9)           |33.25      |0.039  |9.7M   |
|monolayer(rec18)          |32.86      |0.048  |9.7M   |

## Recurrent Experiment(with distance)
recurrent=3
|Model                     |BLEU       |Loss   |Dist  |
|----                      |----       |----   |----  |
|tansformer(layer=6)       |34.65      |-      |-     |
|teach(0-6layer)           |33.27      |0.039  |5.09  |
|teach(0-5layer)           |33.84      |0.060  |9.05  |
|teach(0-4layer)           |34.19      |0.045  |6.88  |
|teach(0-3layer)           |34.38      |0.028  |5.39  |
|teach(0-2layer)           |34.62      |0.016  |4.13  |
|teach(0-1layer)           |34.69      |0.004  |2.91  |

|Model                     |BLEU       |Loss   |Dist  |
|----                      |----       |----   |----  |
|tansformer(layer=6)       |34.65      |-      |-     |
|teach(0-6layer)           |33.27      |0.039  |5.09  |
|teach(1-6layer)           |33.69      |0.033  |4.45  |
|teach(2-6layer)           |33.76      |0.027  |3.70  |
|teach(3-6layer)           |33.97      |0.021  |3.00  |
|teach(4-6layer)           |34.38      |0.012  |3.08  |
|teach(5-6layer)           |34.68      |0.001  |4.56  |


# wmt16 BLEU Scores

## Transformer(base) Experiment(en=>de)

*deprecated*
|Model                           |BLEU       |Param   |
|----                            |----       |----    |
|tansformer(base) +large batch   |26.83      |60.9M   |
|layer=(10,3) +rec(L*) +large batch|27.94    |60.9M   |
|layer=(10,3) +rec(L*)           |Failed     |60.9M   |
|&emsp;+prenorm                  |27.49      |60.9M   |
|layer=(10,3) +rec(FGL*)         |26.95      |60.9M   |
|&emsp;+prenorm                  |Failed     |60.9M   |
|DLCL(30)                        |Failed     |120.8M  |
|&emsp;+prenorm                  |28.06      |120.8M  |
|&emsp;&emsp;+layer batch        |28.25      |120.8M  |

average last 5 checkpoints, length penalty 0.6
|Model                           |BLEU       |Sacrebleu  |Param   |
|----                            |----       |----       |----    |
|tansformer(base)                |27.19      |27.1       |60.9M   |
|DLCL(25)+prenorm                |28.45      |28.3       |120.8M  |
|DLCL(30)+prenorm                |28.18      |28.1       |136.6M  |
|layer(10,3) +rec +dlcl +prenorm |27.52      |27.4       |60.9M   |

seed:2
dlcl
   prenorm, small batch, lr 2e-3, warmup 16000 
|Model   |result|
|----    |----|
|recurrent + dlcl|-|
|&emsp;prenorm, small batch, lr 1e-3, warmup 8000  |failed  |
|&emsp;prenorm, small batch, lr 7e-r, warmup 16000 |27.52   |
|&emsp;small batch, lr 7e-4, warmup 16000          |bleu diverged|
|recurrent + fgl|-|
|&emsp;prenorm, small batch, lr 7e-4, warmup 16000 |failed|
|&emsp;prenorm, large batch, lr 7e-4, warmup 16000 |failed|
|&emsp;prenorm, large batch, lr 7e-4, warmup 8000  |failed|
|&emsp;small batch, batch, lr 7e-4, warmup 16000   |failed|
|&emsp;small batch, batch, lr 7e-4, warmup 8000    |bleu diverged|
recurrent + dlcl
   prenorm, small batch, lr 1e-3, warmup 8000: failed
   prenorm, small batch, lr 7e-r, warmup 16000: 100k is best
   small batch, lr 7e-4, warmup 16000: diverged after bleu reach 26
recurrent + fgl
   prenorm, small batch, lr 7e-4, warmup 16000: failed
   prenorm, large batch, lr 7e-4, warmup 16000: failed
   prenorm, large batch, lr 7e-4, warmup 8000: failed
   small batch, batch, lr 7e-4, warmup 16000: failed
   small batch, batch, lr 7e-4, warmup 8000: diverged after bleu reach 20
   


   


# Distance

|Layer   |0    |2    |4    |6    |
|----    |---- |---- |---- |---- |
|0       |1    |0.59 |0.33 |0.24 |
|2       |0.59 |1    |0.87 |0.73 |
|4       |0.33 |0.87 |1    |0.95 |
|6       |0.24 |0.73 |0.95 |1    |

|Layer   |0    |2    |4    |6    |
|----    |---- |---- |---- |---- |
|0       |0    |4.13 |6.88 |5.09 |
|2       |4.13 |0    |3.74 |3.70 |
|4       |6.88 |3.74 |0    |3.08 |
|6       |5.09 |3.70 |3.08 |0    |

|Layer   |0    |1    |2    |3    |4    |5    |6    |
|----    |---- |---- |---- |---- |---- |---- |---- |
|0       |1    |0.76 |0.59 |0.45 |0.33 |0.24 |0.24 |
|1       |0.76 |1    |0.94 |0.83 |0.70 |0.58 |0.54 |
|2       |0.59 |0.94 |1    |0.96 |0.87 |0.77 |0.73 |
|3       |0.45 |0.83 |0.96 |1    |0.97 |0.91 |0.87 |
|4       |0.33 |0.70 |0.87 |0.97 |1    |0.98 |0.95 |
|5       |0.24 |0.58 |0.77 |0.91 |0.98 |1    |0.99 |
|6       |0.24 |0.54 |0.73 |0.87 |0.95 |0.99 |1    |

|Layer   |0    |1    |2    |3    |4    |5    |6    |
|----    |---- |---- |---- |---- |---- |---- |---- |
|0       |0    |2.91 |4.13 |5.39 |6.88 |9.05 |5.09 |
|1       |2.91 |0    |1.75 |3.42 |5.20 |7.60 |4.45 |
|2       |4.13 |1.75 |0    |1.86 |3.74 |6.25 |3.70 |
|3       |5.39 |3.42 |1.86 |0    |1.96 |4.58 |3.00 |
|4       |6.88 |5.20 |3.74 |1.96 |0    |2.72 |3.08 |
|5       |9.05 |7.60 |6.25 |4.58 |2.72 |0    |4.56 |
|6       |5.09 |4.45 |3.70 |3.00 |3.08 |4.56 |0    |

<!-- 
[[1.         0.75872105 0.59497962 0.44931386 0.32726353 0.2373586 0.24223235]
 [0.75872105 1.         0.94305359 0.8254143  0.69868575 0.57894772 0.54053244]
 [0.59497962 0.94305359 1.         0.95573642 0.8702652  0.76729875 0.72521693]
 [0.44931386 0.8254143  0.95573642 1.         0.97284468 0.90633412 0.86864171]
 [0.32726353 0.69868575 0.8702652  0.97284468 1.         0.9754058 0.94743183]
 [0.2373586  0.57894772 0.76729875 0.90633412 0.9754058  1.        0.98842898]
 [0.24223235 0.54053244 0.72521693 0.86864171 0.94743183 0.98842898 1.        ]]
  -->

<!-- 
[[0.         2.91455636 4.13421921 5.38820975 6.88078016 9.04587496 5.09184072]
 [2.91455636 0.         1.75180434 3.42433957 5.20255742 7.59693469 4.45318277]
 [4.13421921 1.75180434 0.         1.86480351 3.74022815 6.24903778 3.69915702]
 [5.38820975 3.42433957 1.86480351 0.         1.96236743 4.57872082 3.00266189]
 [6.88078016 5.20255742 3.74022815 1.96236743 0.         2.71967691 3.08481567]
 [9.04587496 7.59693469 6.24903778 4.57872082 2.71967691 0.         4.56293378]
 [5.09184072 4.45318277 3.69915702 3.00266189 3.08481567 4.56293378 0.        ]]
  -->


<!-- # [[1.         0.59497962 0.32726353 0.24223235]
#  [0.59497962 1.         0.8702652  0.72521693]
#  [0.32726353 0.8702652  1.         0.94743183]
#  [0.24223235 0.72521693 0.94743183 1.        ]]
# [[0.         4.13421921 6.88078016 5.09184072]
#  [4.13421921 0.         3.74022815 3.69915702]
#  [6.88078016 3.74022815 0.         3.08481567]
#  [5.09184072 3.69915702 3.08481567 0.        ]] -->