# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F
import logging as lgg
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam, WarmupCosineSchedule
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, train_loader, dev_loader, test_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.lr,
                        warmup=args.warmup_ratio,
                        t_total=len(train_loader) * args.epoch)

    total_batch = 0
    dev_best_acc_top = 0.0
    dev_best_acc_sec = 0.0
    dev_best_acc_conn = 0.0
    dev_best_f1_top = 0.0
    dev_best_f1_sec = 0.0
    dev_best_f1_conn = 0.0

    last_improve = 0
    flag = False
    for epoch in range(args.epoch):
        start_time = time.time()
        lgg.info('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        for i, (x, _, mask, _, y1_top, y1_sec, y1_conn, _, _, _, arg1_mask, arg2_mask) in enumerate(train_loader):
            model.train()
            logits_top, logits_sec, logits_conn, loss = model(x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask, train=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_batch += 1
            if total_batch % args.evaluate_steps == 0:
                print(total_batch)

                y_true_top = y1_top.data.cpu() # (batch)
                y_true_sec = y1_sec.data.cpu() # (batch)
                y_true_conn = y1_conn.data.cpu() # (batch)
                
                y_predit_top = torch.max(logits_top.data, 1)[1].cpu() # (batch)
                y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu() # (batch)
                y_predit_conn = torch.max(logits_conn.data, 1)[1].cpu() # (batch)
                
                train_acc_top = metrics.accuracy_score(y_true_top, y_predit_top)
                train_acc_sec = metrics.accuracy_score(y_true_sec, y_predit_sec)
                train_acc_conn = metrics.accuracy_score(y_true_conn, y_predit_conn)
                
                # evaluate
                loss_dev, acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn = evaluate(args, model, dev_loader)
                
                if (acc_top + acc_sec + acc_conn + f1_top + f1_sec + f1_conn) \
                    > (dev_best_acc_top+dev_best_acc_sec+dev_best_acc_conn+dev_best_f1_top+dev_best_f1_sec+dev_best_f1_conn):
                    dev_best_f1_top = f1_top
                    dev_best_f1_sec = f1_sec
                    dev_best_f1_conn = f1_conn
                    dev_best_acc_top = acc_top
                    dev_best_acc_sec = acc_sec
                    dev_best_acc_conn = acc_conn
                    torch.save(model.state_dict(), args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)

                msg = 'top-down:TOP: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                        'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(total_batch, loss.item(), train_acc_top, loss_dev, acc_top, f1_top, time_dif, improve))
                msg = 'top-down:SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                        'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(total_batch, loss.item(), train_acc_sec, loss_dev, acc_sec, f1_sec, time_dif, improve))
                msg = 'top-down:CONN: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                        'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(total_batch, loss.item(), train_acc_conn, loss_dev, acc_conn, f1_conn, time_dif, improve))

                lgg.info(' ')
                lgg.info(' ')

                if total_batch - last_improve > args.require_improvement:
                    # training stop
                    lgg.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
        if flag:
            break

        time_dif = get_time_dif(start_time)
        lgg.info("Train time usage: {}".format(time_dif))
        acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
            = test(args, model, test_loader)

    dev_msg = 'dev_best_acc_top: {0:>6.2%},  dev_best_f1_top: {1:>6.2%}, \n' +\
                'dev_best_acc_sec: {2:>6.2%},  dev_best_f1_sec: {3:>6.2%}, \n' +\
                'dev_best_acc_conn: {4:>6.2%},  dev_best_f1_conn: {5:>6.2%}'
    lgg.info(dev_msg.format(dev_best_acc_top, dev_best_f1_top,
                            dev_best_acc_sec, dev_best_f1_sec,
                            dev_best_acc_conn, dev_best_f1_conn))


def test(args, model, test_loader):
    model.load_state_dict(torch.load(args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt'))
    model.eval()
    start_time = time.time()

    test_loss, acc_top, f1_top, report_top, confusion_top, \
    acc_sec, f1_sec, report_sec, confusion_sec, acc_conn, f1_conn, \
    consistency_top_sec, consistency_sec_conn, consistency_top_sec_conn = \
        evaluate(args, model, test_loader, test=True)

    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'TOP: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_top, f1_top))
    msg = 'SEC: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_sec, f1_sec))
    msg = 'CONN: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_conn, f1_conn))
    msg = 'consistency_top_sec: {0:>6.2%},  consistency_sec_conn: {1:>6.2%}, consistency_top_sec_conn: {2:>6.2%}'
    lgg.info(msg.format(consistency_top_sec, consistency_sec_conn, consistency_top_sec_conn))
    lgg.info(report_top)
    lgg.info(report_sec)
    return acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn


def evaluate(args, model, data_loader, test=False):
    model.eval()
    loss_total = 0
    predict_all_top = np.array([], dtype=int)
    labels1_all_top = np.array([], dtype=int)
    labels2_all_top = np.array([], dtype=int)

    predict_all_sec = np.array([], dtype=int)
    labels1_all_sec = np.array([], dtype=int)
    labels2_all_sec = np.array([], dtype=int)

    predict_all_conn = np.array([], dtype=int)
    labels1_all_conn = np.array([], dtype=int)
    labels2_all_conn = np.array([], dtype=int)

    with torch.no_grad():
        for i, (x, _, mask, _, y1_top, y1_sec, y1_conn, y2_top, y2_sec, y2_conn, arg1_mask, arg2_mask) in enumerate(data_loader):

            logits_top, logits_sec, logits_conn = model(x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask, train=False)
            
            loss_top = F.cross_entropy(logits_top, y1_top)
            loss_sec = F.cross_entropy(logits_sec, y1_sec)
            loss_conn = F.cross_entropy(logits_conn, y1_conn)

            loss = loss_top + loss_sec + loss_conn
            loss_total += loss
            
            y_predit_top = torch.max(logits_top.data, 1)[1].cpu().numpy()
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu().numpy()
            y_predit_conn = torch.max(logits_conn.data, 1)[1].cpu().numpy()

            y1_true_top = y1_top.data.cpu().numpy()
            y2_true_top = y2_top.data.cpu().numpy()
            labels1_all_top = np.append(labels1_all_top, y1_true_top) # collect all top true label
            labels2_all_top = np.append(labels2_all_top, y2_true_top)
            predict_all_top = np.append(predict_all_top, y_predit_top) # collect all top predicted label

            y1_true_sec = y1_sec.data.cpu().numpy()
            y2_true_sec = y2_sec.data.cpu().numpy()
            labels1_all_sec = np.append(labels1_all_sec, y1_true_sec) # collect all sec true label
            labels2_all_sec = np.append(labels2_all_sec, y2_true_sec)
            predict_all_sec = np.append(predict_all_sec, y_predit_sec) # collect all sec predicted label

            y1_true_conn = y1_conn.data.cpu().numpy()
            y2_true_conn = y2_conn.data.cpu().numpy()
            labels1_all_conn = np.append(labels1_all_conn, y1_true_conn) # collect all conn true label
            labels2_all_conn = np.append(labels2_all_conn, y2_true_conn)
            predict_all_conn = np.append(predict_all_conn, y_predit_conn) # collect all conn predicted label

    predict_sense_top = predict_all_top
    gold_sense_top = labels1_all_top
    mask = (predict_sense_top == labels2_all_top)
    gold_sense_top[mask] = labels2_all_top[mask]

    predict_sense_sec = predict_all_sec
    gold_sense_sec = labels1_all_sec
    mask = (predict_sense_sec == labels2_all_sec)
    gold_sense_sec[mask] = labels2_all_sec[mask]

    predict_sense_conn = predict_all_conn
    gold_sense_conn = labels1_all_conn
    mask = (predict_sense_conn == labels2_all_conn)
    gold_sense_conn[mask] = labels2_all_conn[mask]

    # PDTB2.0 cutoff
    if test:
        cut_off = 1039
    else:
        cut_off = 1165
    
    # PDTB3.0 cutoff    
    # if test:
    #     cut_off = 1474
    # else:
    #     cut_off = 1653

    acc_top = metrics.accuracy_score(gold_sense_top, predict_sense_top)
    f1_top = metrics.f1_score(gold_sense_top, predict_sense_top, average='macro')

    gold_sense_sec = gold_sense_sec[: cut_off]
    predict_sense_sec = predict_sense_sec[: cut_off]
    acc_sec = metrics.accuracy_score(gold_sense_sec, predict_sense_sec)
    f1_sec = metrics.f1_score(gold_sense_sec, predict_sense_sec, average='macro')

    acc_conn = metrics.accuracy_score(gold_sense_conn, predict_sense_conn)
    f1_conn = metrics.f1_score(gold_sense_conn, predict_sense_conn, average='macro')

    if test:
        report_top = metrics.classification_report(gold_sense_top, predict_sense_top, target_names=args.i2top, digits=4)
        confusion_top = metrics.confusion_matrix(gold_sense_top, predict_sense_top)

        report_sec = metrics.classification_report(gold_sense_sec, predict_sense_sec, target_names=args.i2sec, digits=4)
        confusion_sec = metrics.confusion_matrix(gold_sense_sec, predict_sense_sec)
        
        #######################################################################################
        consistency_top_sec = 0
        for i in range(cut_off):
            if gold_sense_top[i] == predict_sense_top[i] and gold_sense_sec[i] == predict_sense_sec[i]:
                consistency_top_sec += 1
        consistency_top_sec = consistency_top_sec / cut_off
        
        consistency_sec_conn = 0
        for i in range(cut_off):
            if gold_sense_sec[i] == predict_sense_sec[i] and gold_sense_conn[i] == predict_sense_conn[i]:
                consistency_sec_conn += 1
        consistency_sec_conn = consistency_sec_conn / cut_off
        
        consistency_top_sec_conn = 0
        for i in range(cut_off):
            if gold_sense_top[i] == predict_sense_top[i] and gold_sense_sec[i] == predict_sense_sec[i] and gold_sense_conn[i] == predict_sense_conn[i]:
                consistency_top_sec_conn += 1
        consistency_top_sec_conn = consistency_top_sec_conn / cut_off
        #######################################################################################

        return loss_total / len(data_loader), acc_top, f1_top, report_top, confusion_top, \
                acc_sec, f1_sec, report_sec, confusion_sec, acc_conn, f1_conn, \
                consistency_top_sec, consistency_sec_conn, consistency_top_sec_conn
                
    return loss_total / len(data_loader), acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn
