# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

from pdtb2 import CorpusReader
import argparse
import os
import sys


top_senses = set(['Temporal', 'Comparison', 'Contingency', 'Expansion'])
selected_second_senses = set([
    'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
    'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
    'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
    'Expansion.Alternative', 'Expansion.List'
])


def arg_filter(input):
    arg = []
    pos = []
    for w in input:
        if w[1].find('-') == -1:
            arg.append(w[0].replace('\/', '/'))
            pos.append(w[1])
    return arg, pos


def preprocess(splitting):
    # following Ji, for 4-way and 11-way
    if splitting == 1:
        train_sec = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                     '12', '13', '14', '15', '16', '17', '18', '19', '20']
        dev_sec = ['00', '01']
        test_sec = ['21', '22']

    # following Lin, for 4-way and 11-way
    elif splitting == 2:
        train_sec = [
            '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
        ]
        dev_sec = ['22']
        test_sec = ['23']

    # instances in 'selected_second_senses'
    arg1_train = []
    arg2_train = []
    sense1_train = []    # top, second, connective
    sense2_train = []    # None, None, None

    arg1_dev = []
    arg2_dev = []
    sense1_dev = []
    sense2_dev = []

    arg1_test = []
    arg2_test = []
    sense1_test = []
    sense2_test = []

    os.chdir(sys.path[0])
    for corpus in CorpusReader('./raw/pdtb2.csv').iter_data():
        if corpus.Relation != 'Implicit':
            continue
        sense_split = corpus.ConnHeadSemClass1.split('.')
        sense_l2 = '.'.join(sense_split[0:2])

        if sense_l2 in selected_second_senses:
            arg1, pos1 = arg_filter(corpus.arg1_pos(wn_format=True))
            arg2, pos2 = arg_filter(corpus.arg2_pos(wn_format=True))
            if corpus.Section in train_sec:
                arg1_train.append(arg1)
                arg2_train.append(arg2)
                sense1_train.append([sense_split[0], sense_l2, corpus.Conn1])
                sense2_train.append([None, None, None])
            elif corpus.Section in dev_sec:
                arg1_dev.append(arg1)
                arg2_dev.append(arg2)
                sense1_dev.append([sense_split[0], sense_l2, corpus.Conn1])
            elif corpus.Section in test_sec:
                arg1_test.append(arg1)
                arg2_test.append(arg2)
                sense1_test.append([sense_split[0], sense_l2, corpus.Conn1])
            else:
                continue

            if corpus.Conn2 is not None:
                sense_split = corpus.Conn2SemClass1.split('.')
                sense_l2 = '.'.join(sense_split[0:2])
                if sense_l2 in selected_second_senses:
                    if corpus.Section in train_sec:
                        arg1_train.append(arg1)
                        arg2_train.append(arg2)
                        sense1_train.append([sense_split[0], sense_l2, corpus.Conn2])
                        sense2_train.append([None, None, None])
                    elif corpus.Section in dev_sec:
                        sense2_dev.append([sense_split[0], sense_l2, corpus.Conn2])
                    elif corpus.Section in test_sec:
                        sense2_test.append([sense_split[0], sense_l2, corpus.Conn2])
            else:
                if corpus.Section in dev_sec:
                    sense2_dev.append([None, None, None])
                elif corpus.Section in test_sec:
                    sense2_test.append([None, None, None])

    assert len(arg1_train) == len(arg2_train) == len(sense1_train) == len(sense2_train)
    assert len(arg1_dev) == len(arg2_dev) == len(sense1_dev) == len(sense2_dev)
    assert len(arg1_test) == len(arg2_test) == len(sense1_test) == len(sense2_test)
    print('train size:', len(arg1_train))
    print('dev size:', len(arg1_dev))
    print('test size:', len(arg1_test))

    if splitting == 1:
        pre = './PDTB/Ji//data//'
    elif splitting == 2:
        pre = './PDTB/Lin//data//'

    with open(pre + 'train.txt', 'w') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_train, arg2_train, sense1_train, sense2_train):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, ' '.join(arg1), ' '.join(arg2)), file=f)
    with open(pre + 'dev.txt', 'w') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_dev, arg2_dev, sense1_dev, sense2_dev):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, ' '.join(arg1), ' '.join(arg2)), file=f)
    with open(pre + 'test.txt', 'w') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_test, arg2_test, sense1_test, sense2_test):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, ' '.join(arg1), ' '.join(arg2)), file=f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='func', choices=['pre', 'test'], type=str, default='pre')
    parser.add_argument('-s', dest='splitting', choices=[1, 2], type=int, default='1')   # 1 for 'Ji', 2 for 'Lin'
    A = parser.parse_args()
    preprocess(A.splitting)

