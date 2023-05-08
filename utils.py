# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


def get_time_dif(start_time):
    """ """
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))
    
    
class MyDataset(Dataset):
    
    def __init__(self, args, path):
        content = self.load_dataset(args, path)
        self.len = len(content)
        self.device = args.device
        
        self.x, \
        self.seq_len, self.mask, self.token_type, \
        self.y1_top, self.y1_sec, self.y1_conn, \
        self.y2_top, self.y2_sec, self.y2_conn, \
        self.arg1_mask, self.arg2_mask = self._to_tensor(content)


    def __getitem__(self, index):
        return self.x[index], \
        self.seq_len[index], self.mask[index], self.token_type[index], \
        self.y1_top[index], self.y1_sec[index], self.y1_conn[index], \
        self.y2_top[index], self.y2_sec[index], self.y2_conn[index], \
        self.arg1_mask[index], self.arg2_mask[index]


    def __len__(self):             
        return self.len


    def load_dataset(self, args, path):
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                labels1, labels2, arg1, arg2 = [_.strip() for _ in lin.split('|||')]
                labels1, labels2 = eval(labels1), eval(labels2)
                labels1[0] = args.top2i[labels1[0]] if labels1[0] is not None else -1
                labels1[1] = args.sec2i[labels1[1]] if labels1[1] is not None else -1
                labels1[2] = args.conn2i[labels1[2]] if labels1[2] is not None else -1
                labels2[0] = args.top2i[labels2[0]] if labels2[0] is not None else -1
                labels2[1] = args.sec2i[labels2[1]] if labels2[1] is not None else -1
                labels2[2] = args.conn2i[labels2[2]] if labels2[2] is not None else -1


                arg1_token = args.tokenizer.tokenize(arg1)
                arg2_token = args.tokenizer.tokenize(arg2)
                token = [CLS] + arg1_token + [SEP] + arg2_token + [SEP]

                token_type_ids = [0] * (len(arg1_token) + 2) + [1] * (len(arg2_token) + 1)
                arg1_mask = [1] * (len(arg1_token) + 2)
                arg2_mask = [0] * (len(arg1_token) + 2) + [1] * (len(arg2_token) + 1)

                input = args.tokenizer(arg1, arg2, truncation=True, max_length=args.pad_size, padding='max_length')
                input_ids = input['input_ids']
                attention_mask = input['attention_mask']
                seq_len = len(token)

                if args.pad_size:
                    if len(token) < args.pad_size:
                        token_type_ids += ([0] * (args.pad_size - len(token)))
                    else:
                        token_type_ids = token_type_ids[:args.pad_size]
                        seq_len = args.pad_size

                    if len(arg1_mask) < args.pad_size:
                        arg1_mask += [0] * (args.pad_size - len(arg1_mask))
                    else:
                        arg1_mask = arg1_mask[:args.pad_size]
                    if len(arg2_mask) < args.pad_size:
                        arg2_mask += [0] * (args.pad_size - len(arg2_mask))
                    else:
                        arg2_mask = arg2_mask[:args.pad_size]
                contents.append((input_ids, seq_len, attention_mask, token_type_ids,
                                 labels1[0], labels1[1], labels1[2],
                                 labels2[0], labels2[1], labels2[2],
                                 arg1_mask, arg2_mask))
        return contents
    
    
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        token_type = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        y1_top = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        y1_sec = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        y1_conn = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        y2_top = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        y2_sec = torch.LongTensor([_[8] for _ in datas]).to(self.device)
        y2_conn = torch.LongTensor([_[9] for _ in datas]).to(self.device)

        arg1_mask = torch.LongTensor([_[10] for _ in datas]).to(self.device)
        arg2_mask = torch.LongTensor([_[11] for _ in datas]).to(self.device)

        return x, seq_len, mask, token_type, y1_top, y1_sec, y1_conn, y2_top, y2_sec, y2_conn, arg1_mask, arg2_mask
