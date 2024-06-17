import argparse
import os



def args_fn():
    parser = argparse.ArgumentParser(description='preprocess label data')

    parser.add_argument('--src', type=str, required=True, help="the source file name")
    parser.add_argument('--tgt', type=str, required=True, help='the target file name')
    parser.add_argument('--input_dir', type=str, default='.', help='the dir of source and target files.')
    parser.add_argument('--output_dir', type=str, default='.', help="the output dir for preocessed file.")
    
    args = parser.parse_args()
    return args


def read_file(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    res = [line.strip().split('\t') for line in lines]
    res = {e[0]:e[1] for e in res}
    return res

def label_data(src, tgt):
    res = []
    for item_id in list(src.keys()):
        question = src[item_id]
        ground_truth = tgt[item_id]
        
        res_line = []
        res_line.append(item_id)
        
        for i in range(len(question)):
            if question[i] != ground_truth[i]:
                res_line.append(str(i+1))
                res_line.append(ground_truth[i])
        if len(res_line) == 1:
            res_line.append('0')
        res.append(', '.join(res_line))
    return res

if __name__ == "__main__":
    args = args_fn()
    
    src_path = os.path.join(args.input_dir, args.src)
    tgt_path = os.path.join(args.input_dir, args.tgt)
    
    src = read_file(src_path)
    tgt = read_file(tgt_path)
    assert len(src) == len(tgt)
    data = label_data(src, tgt)
    print(f"len data:{len(data)}")

    f_name = 'label.lbl'+ '.tsv'
    print(f_name)
    with open (os.path.join(args.output_dir, f_name), 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')


# python -u preprocess_label.py --src src.txt --tgt tgt.txt 