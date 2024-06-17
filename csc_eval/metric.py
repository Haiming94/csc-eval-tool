import os

from csc_eval.metric_core import metric_file
from csc_eval.remove_de import remove_de


class Metric:

    def __init__(self, ):
        pass
        

    def load_tsv(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            print(file)
            lines = f.readlines()
        res = [line.strip().split('\t') for line in lines]
        res = {e[0]:e[1] for e in res}
        return res

    def metric(self, src_path, pred_path, pred_lbl_path, label_path, should_remove_de=False):
        src = self.load_tsv(src_path)
        pred = self.load_tsv(pred_path)
        
        self.write_pred(src, pred, pred_lbl_path)
        if should_remove_de:
            remove_de(
                input_path=pred_lbl_path,
                output_path=pred_lbl_path,
            )
        scores = metric_file(
            pred_path=pred_lbl_path,
            targ_path=label_path,
            #do_char_metric=False,
        )
        # ns, vs = [], []
        # for k,v in scores.items():
        #     scores[k] = round(v, 4)
        #     print(k, round(v, 2))
        #     ns.append(k)
        #     vs.append(str(round(v, 2)))
        # print(" | ".join(ns))
        # print(" | ".join(vs))
        return scores

    def write_pred(self, src_dict, pred_dict, pred_lbl_path):
        pred_lbl_list = []
        
        for item_id in src_dict.keys():
            src = src_dict[item_id]
            pred = pred_dict[item_id]
            pred_lbl = self.process_item(src, pred, item_id)
            pred_lbl_list.append(pred_lbl)

        pred_dir = os.path.dirname(pred_lbl_path)
        os.makedirs(pred_dir, exist_ok=True)

        with open(pred_lbl_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pred_lbl_list))
        print('\n\n')
        print(f'Metric write to "{pred_lbl_path}"')

    def process_item(self, src, pred, item_id):
        
        if len(src) < len(pred):
            pred = pred[:len(src)]

        if len(src) > len(pred):
            src = src[:len(pred)]
        assert len(pred) == len(src)
        
        item = [item_id]
        for i, (a, b) in enumerate(zip(src, pred), start=1):
            if a != b:
                item.append(str(i))
                item.append(b)
        if len(item) == 1:
            item.append('0')
        pred_lbl = ', '.join(item)

        return pred_lbl

