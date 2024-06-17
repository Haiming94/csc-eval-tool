from csc_eval import Metric


src_file = 'data/src.txt'
tgt_file = 'data/tgt.txt'
pred_file = 'data/pred.txt'
lbl_file = 'data/label.lbl.tsv'

pred_lbl_file = 'data/pred.lbl.tsv'
metricor = Metric()
metricor.metric(src_file, pred_file, pred_lbl_file, lbl_file)