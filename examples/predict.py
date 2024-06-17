from transformers import GPT2Tokenizer,GPT2LMHeadModel
from transformers import BertTokenizer
import torch
import pytorch_lightning as pl
from pypinyin import pinyin,Style

import os
import json
import argparse
from transformers import BertConfig
from models.modeling_multitask import Dynamic_GlyceBertForMultiTask
from tokenizers import BertWordPieceTokenizer
from torch.nn import functional as F
# from flask_cors import cross_origin

#model_path = "IDEA-CCNL/Wenzhong-GPT2-110M"
bert_path = "/home/hm/spaces/zsy/CSC-Eval/examples/ckpt/FPT"
ckpt_path = "/home/hm/spaces/zsy/CSC-Eval/examples/ckpt/bs32epoch30/epoch=25-df=78.9946-cf=77.1993.ckpt"

vocab_path = os.path.join(bert_path, "vocab.txt")
tokenizer = BertWordPieceTokenizer(vocab_path, lowercase = True)
# tokenizer = BertWordPieceTokenizer.from_pretrained(vocab_path)
model = Dynamic_GlyceBertForMultiTask.from_pretrained(bert_path)



if ckpt_path is not None:
    print("loading from ", ckpt_path)
    ckpt = torch.load(ckpt_path,)["state_dict"]
    new_ckpt = {}
    for key in ckpt.keys():
        new_ckpt[key[6:]] = ckpt[key]
    model.load_state_dict(new_ckpt,strict=False)
    print(model.device, torch.cuda.is_available())
    #vocab_size = bert_config.vocab_size

class hanzi2pinyin():

    def __init__(self, chinese_bert_path, max_length: int = 512):
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        self.config_path = os.path.join(chinese_bert_path, 'config')
        self.max_length = max_length

        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)

        
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

token2pinyin = hanzi2pinyin(bert_path)

def get_input_idx(input_text):
    encoded = tokenizer.encode(input_text)
    input_ids = torch.tensor(encoded.ids).unsqueeze(0)
    pinyin_ids = torch.tensor(token2pinyin.convert_sentence_to_pinyin_ids(input_text, encoded)).unsqueeze(0)
    return input_ids, pinyin_ids

def forward(input_ids, pinyin_ids, labels=None, pinyin_labels=None, tgt_pinyin_ids=None, var=1):
    """"""
    # print(input_ids)
    attention_mask = (input_ids != 0).long()
    # print(attention_mask)
    return model(
        input_ids,
        pinyin_ids,
        attention_mask=attention_mask,
        labels=labels,
        tgt_pinyin_ids=tgt_pinyin_ids, 
        pinyin_labels=pinyin_labels,
        gamma=0,
    )


# tokenizer,model = load_model(bert_path, ckpt_path)

def main(input_text: str, max_length: int = 256, top_p: float = 0.6, num_return_sequences: int = 5):
    # print(input_text)
    
    # inputs = tokenizer(input_text,return_tensors='pt')
    #inputs = get_input_idx(input_text)
    input_ids, pinyin_ids = get_input_idx(input_text)
    # print(input_ids)
    mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
    with torch.no_grad():
        outputs = forward(input_ids, pinyin_ids).logits
        # outputs = model(**inputs).logits

        predict_scores = F.softmax(outputs, dim=-1)
        predict_labels = (torch.argmax(predict_scores, dim=-1) * mask).cpu()

        # print(type(predict_labels.squeeze()))
        
        pred_txt = tokenizer.decode(predict_labels.squeeze().tolist())
        
        predict_text = ''.join(pred_txt.split(' '))
        # print(predict_text)
        
        return predict_text
    
if __name__ == '__main__':
    

    with open ('/home/hm/spaces/zsy/CSC-Eval/data/src.txt', 'r') as f:
        data = f.readlines()

    for line in data:
        res = main(line)
        with open ('/home/hm/spaces/zsy/CSC-Eval/data/pred.txt', 'a') as f:
            # for line in res:
            f.write(res)
            f.write('\n')

# python -u predict.py --src src.txt --pred pred.txt 