# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import time
import torch
import argparse
import logging as lgg

import transformers.utils.logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datetime import datetime
import warnings
import numpy as np
import os
import random

from training import train
from GOLF import Model
from utils import MyDataset, get_time_dif

warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()


def setlogging(level, filename):
    for handler in lgg.root.handlers[:]:
        lgg.root.removeHandler(handler)
    lgg.basicConfig(level=level,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M',
                    filename=filename,
                    filemode='w')
    logc = lgg.StreamHandler()
    logc.setLevel(level=lgg.DEBUG)
    logc.setFormatter(lgg.Formatter('%(message)s'))
    lgg.getLogger().addHandler(logc)


def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Global and Local Hierarchy-aware Contrastive Framework for Hierarchical Implicit Discourse Relation Recognition')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--data_file', type=str, default='PDTB/Ji/data/', help='the file of data')
    parser.add_argument('--log_file', type=str, default='PDTB/Ji/log/', help='the file of log')
    parser.add_argument('--save_file', type=str, default='PDTB/Ji/saved_dict/', help='save model file')
    
    ## model arguments
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='the name of pretrained model')
    parser.add_argument('--freeze_bert', action='store_true', default=False, help='whether freeze the parameters of bert')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of contrastive learning')
    parser.add_argument('--num_co_attention_layer', type=int, default=2, help='number of co-attention layers')
    parser.add_argument('--num_gcn_layer', type=int, default=2, help='number of gcn layers')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='dropout rate after gcn layer')
    parser.add_argument('--label_embedding_size', type=int, default=100, help='embedding dimension of labels')
    parser.add_argument('--lambda_global', type=float, default=0.1, help='lambda for global_hierarcial_contrastive_loss')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='lambda for local_hierarcial_contrastive_loss')
    ## training arguments
    parser.add_argument('--pad_size', type=int, default=100, help='the max sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=15, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='warmup_ratio')
    parser.add_argument('--evaluate_steps', type=int, default=100, help='number of evaluate_steps')
    parser.add_argument('--require_improvement', type=int, default=10000, help='early stop steps')
    
    args = parser.parse_args()

    
    args.i2top = [x.strip() for x in open(args.data_file + 'top.txt').readlines()]
    args.top2i = dict((x, xid) for xid, x in enumerate(args.i2top))
    args.n_top = len(args.i2top)
    args.i2sec = [x.strip() for x in open(args.data_file + 'sec.txt').readlines()]
    args.sec2i = dict((x, xid) for xid, x in enumerate(args.i2sec))
    args.n_sec = len(args.i2sec)
    args.i2conn = [x.strip() for x in open(args.data_file + 'conn.txt').readlines()]
    args.conn2i = dict((x, xid) for xid, x in enumerate(args.i2conn))
    args.n_conn = len(args.i2conn)
    args.label_num = args.n_top + args.n_sec + args.n_conn # total label num(top:4,second:11,conn:102)
    
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    args.config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    args.t = datetime.now().strftime('%B%d-%H:%M:%S')
    args.log = args.log_file + str(args.t) +'.log'
    print(args.log)
    args.device = torch.device('cuda:{0}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    
    setlogging(lgg.DEBUG, args.log)
    seed_torch(args.seed)

    hyper_parameters = args.__dict__.copy()
    hyper_parameters['i2conn'] = ''
    hyper_parameters['conn2i'] = ''
    hyper_parameters['i2top'] = ''
    hyper_parameters['top2i'] = ''
    hyper_parameters['i2sec'] = ''
    hyper_parameters['sec2i'] = ''
    hyper_parameters['tokenizer'] = ''
    hyper_parameters['config'] = ''
    lgg.info(hyper_parameters)

    start_time = time.time()
    lgg.info("Loading data...")
    
    train_dataset = MyDataset(args, args.data_file + 'train.txt')
    dev_dataset = MyDataset(args, args.data_file + 'dev.txt')
    test_dataset = MyDataset(args, args.data_file + 'test.txt')
    
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    
    time_dif = get_time_dif(start_time)
    lgg.info("Time usage: {}".format(time_dif))

    # train
    model = Model(args).to(args.device)
    train(args, model, train_loader, dev_loader, test_loader)
