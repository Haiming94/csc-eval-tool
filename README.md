# CSC-Eval

`CSC-Eval` is a tool for evaluating the performance of Chinese Spelling Check (CSC) algorithms, which includes the evaluation of the accuracy, precision, recall, and F1-score of both chatacter-leval and sentence-level.

## Metric







- Accuracy: 

$$

acc_{character} = \frac{TP_{character}+TN_{character}}{TP_{character}+FN_{character}+TN_{character}+FP_{character}} \\

acc_{sentence} = \frac{TP_{sentence}+TN_{sentence}}{TP_{sentence}+FN_{sentence}+TN_{sentence}+FP_{sentence}}

$$






- Precision:

$$
P_{character} = \frac{TP}{TP+FP}
$$




- Recall:

$$
R_{character} = \frac{TP}{TP+FN}
$$




- F1-score:

$$

F1_{character} = \frac{2*P_{\#}*R_{\#}}{P_{\#}+R_{\#}}  \\
F1_{sentence} = \frac{2*P_{\#}*R_{\#}}{P_{\#}+R_{\#}}

$$


## Packages

- transformers


## How to use this tool

1. Place the input sentences, golden sentences, and model prediction results into the data folder. Please refer to `src.txt`, `tgt.txt`, and `pred.txt` for the format of each file.
2. Run `python preprocess_label.py` to create `label.lbl.tsv`.
3. Run `python preprocess_label.py` to get `pred.txt`.