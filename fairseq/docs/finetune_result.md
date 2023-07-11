

|Model      | BLEU|
|----|----|
|Transfomer(base,dropout=0.4)   |27.93    |
|Transfomer(pretrain)           |28.74    |
|Transfomer(finetune)           |31.85    |
|Transfomer(last)               |31.3x    |


# best finetune
|Positive |Negative|Equal|All|
|----|----|----|----|
|3292|2054|1404|6750|
Avg. Increase BLEU 12.34
Avg. Decrease BLEU 9.59

# last finetune
|Positive |Negative|Equal|All|
|----|----|----|----|
|3239|2263|1248|6750|
Avg. Increase BLEU 12.47
Avg. Decrease BLEU 10.72


# BLEU delta data
|epoch|150|180|210|240|270|300|330|360|390|420|
|----|----|----|----|----|----|----|----|----|----|----|
|positive|12.44|12.28|12.36|12.32|12.28|12.41|12.45|12.48|12.47|12.47|
|negative|10.19|10.21|10.27|10.35|10.42|10.50|10.46|10.63|10.68|10.72|

# n-gram sparse
|n-gram|positive set|negative set|
|----|-----|----|
|1|14.27%|13.55%|
|2|36.02%|35.82%|
|3|64.78%|65.53%|

source
|n-gram|positive set|negative set|
|----|-----|----|
|1|7.33%|7.28%|
|2|26.31%|25.82%|
|3|54.64%|54.76%|


# sim score corrcoef
|dataset|PCCs|src avg sim score|tgt avg sim score|
|----|-----|----|----|
|pos|**0.529**|20.87|9.56|
|neg|**0.356**|21.38|8.72|
pos corrcoef 0.529
src avg sim score: 20.87
tgt avg sim score: 9.56

neg corrcoef 0.356
src avg sim score: 21.38
tgt avg sim score: 8.72

|dataset|PCCs|src avg sim score|tgt avg sim score|
|----|-----|----|----|
|pos|**0.621**|20.87|10.79|
|neg|**0.608**|21.38|10.85|
pos hypo corrcoef 0.621
src avg sim score: 20.87
tgt avg sim score: 10.79

neg hypo corrcoef 0.608
src avg sim score: 21.38
tgt avg sim score: 10.85


[(52, 70.71), (267, 65.25), (60, 57.11), (656, 54.790000000000006), (366, 43.19), (274, 42.73), (2679, 42.05), (739, 41.71), (645, 41.660000000000004), (1333, 41.129999999999995)]
[(114, -100.0), (137, -80.78999999999999), (351, -80.78999999999999), (1013, -67.72), (967, -64.33000000000001), (603, -59.699999999999996), (2406, -56.13), (574, -56.089999999999996), (579, -54.3), (2947, -51.980000000000004)]
Pearson corrcoef:  [[1.         0.29671002]
 [0.29671002 1.        ]]


114
hypo:
Wettlauf in die Römer
tgt:
Tourismus : Abstieg zu den Römern
src:
Tourism : Descent to the Romans

137
hypo:
Das wurde von Sendern SABC berichtet .
tgt:
Dies berichtete der Sender SABC .
src:
This was reported by broadcaster SABC .

351
hypo:
Braucht die Baumschule eine neue Sandbox ?
tgt:
Braucht der Kindergarten einen neuen Sandkasten ?
src:
Does the nursery school need a new sand box ?

1013:
hypo:
Union und SPD haben ihre Koalitionsverhandlungen fortgesetzt , in denen Themen wie innere Angelegenheiten und Gerechtigkeit behandelt wurden .
tgt:
Union und SPD haben ihre Koalitionsverhandlungen mit den Themen Inneres und Justiz fortgesetzt .
src:
Union and SPD have continued their coalition negotiations , addressing the topics of internal affairs and justice .

967
hypo:
Der einzige Rückgriff ist auf Lohnerhöhungen und öffentliche Ausgaben - angetrieben von Berlin .
tgt:
Die einzige Möglichkeit besteht darin , Löhne und öffentliche Ausgaben zu kürzen – angespornt von Berlin .
src:
The only recourse is to slash wages and public spending - spurred on by Berlin .

579
hypo:
Vor allem ist die Art , wie das Team spielt , beeindruckend .
tgt:
Vor allem die Art und Weise , wie die Mannschaft spielt , ist beeindruckend .
src:
Most of all , the manner in which the team is playing is impressive .

2947:
hypo:
Das Leipferdingen Heim wird 40 Jahre alt werden , die Geisingen Schule ist nun seit 50 Jahren an ihrem neuen Standort und wird dies am 10. Mai feiern , und die Polyhymnia Leipferdingen Musikgemeinde wird 150 Jahre alt werden und wird dies als Teil des Brunnenfestes feiern . vom 4. bis 7. Juli .
tgt:
Das Altenwerk Leipferdingen wird 40 Jahre alt , die Geisinger Schule ist seit 50 Jahren am neuen Standort und feiert dies am 10. Mai , der Musikverein Polyhymnia Leipferdingen wird 150 Jahre alt und feiert dies im Rahmen des Brunnenfestes vom 4. bis 7. Juli .
src:
The Leipferdingen Nursing Home will turn 40 years old , the Geisingen School has now been at its new location for 50 years and will be celebrating this on 10 May , and the Polyhymnia Leipferdingen music association will turn 150 years old and will be celebrating this as part of the Brunnenfest &#91; Fountain Festival &#93; from 4 to 7 July .



|dataset|s(gnl) ~s(dest)|~s(gnl) s(dest) |
|----|-----|----|
|pos|0.0289|4.691|
|neg|0.0235|4.900|

|dataset|s(gnl) ~s(dest)|~s(gnl) s(dest) |
|----|-----|----|
|pos|0.471|2.289|
|neg|0.357|2.386|


