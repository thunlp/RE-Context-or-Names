
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from apex import amp
from tqdm import trange
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *



def f1_score(output, label, rel_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)
    return micro_f1, f1_by_relation

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, devBagTest=None, testBagTest=None):
    # total step
    step_tot = len(train_dataloader) * args.max_epoch

    # optimizer
    if args.optim == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)
    elif args.optim == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, args.lr)
    elif args.optim == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, args.lr)

    # amp training
    if args.optim == "adamw":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Data parallel
    model = nn.DataParallel(model)
    model.train()
    model.zero_grad()

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    for i in range(args.max_epoch):
        for batch in train_dataloader:
            inputs = {
                "input_ids":batch[0],
                "mask":batch[1],
                "h_pos":batch[2],
                "t_pos":batch[3],
                "label":batch[4]
            }
            model.training = True
            model.train()
            loss, output = model(**inputs)
            if args.optim == "adamw":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if args.optim == "adamw":
                scheduler.step()
            model.zero_grad()
            global_step += 1

            output = output.cpu().detach().numpy()
            label = batch[4].numpy()
            crr = (output == label).sum()
            tot = label.shape[0]

            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr/tot))
            sys.stdout.flush()

        # dev
        with torch.no_grad():
            print("")
            print("deving....")
            model.training = False
            model.eval()

            if args.dataset == "semeval" or args.dataset == "tacred":
                eval_func = eval_F1
            elif args.dataset == "wiki80" or args.dataset == "chemprot":
                eval_func = eval_ACC
            
            score = eval_func(args, model, dev_dataloader)
            if score > best_dev_score:
                best_dev_score = score
                best_test_score = eval_func(args, model, test_dataloader)
                print("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
            else:
                print("Dev score: %.3f" % score)
            print("-----------------------------------------------------------")         


    print("@RESULT: " + args.dataset +" Test score is %.3f" % best_test_score)
    f = open("../log/re_log", 'a+')
    if args.ckpt_to_load == "None":
        f.write("bert-base\t" + args.dataset + "\t" + str(time.ctime())  +"\n")
    else:
        f.write(args.ckpt_to_load + "\t" + args.dataset + "\t" +str(time.ctime()) +"\n")
    f.write("@RESULT: Best Dev score is %.3f, Test score is %.3f\n" % (best_dev_score, best_test_score))
    f.write("--------------------------------------------------------------\n")
    f.close()


def eval_F1(args, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        inputs = {
            "input_ids":batch[0],
            "mask":batch[1],
            "h_pos":batch[2],
            "t_pos":batch[3],
            "label":batch[4]
        }
        _, output = model(**inputs)
        tot_label.extend(batch[4].tolist())
        tot_output.extend(output.cpu().detach().tolist())

    f1, _ = f1_score(tot_output, tot_label, args.rel_num)     
    return f1

def eval_ACC(args, model, dataloader):
    tot = 0.0
    crr = 0.0
    for batch in dataloader:
        inputs = {
            "input_ids":batch[0],
            "mask":batch[1],
            "h_pos":batch[2],
            "t_pos":batch[3],
            "label":batch[4]
        }
        _, output = model(**inputs)
        output = output.cpu().detach().numpy()
        label = batch[4].numpy()
        crr += (output==label).sum()
        tot += label.shape[0]

        sys.stdout.write("acc: %.3f\r" % (crr/tot)) 
        sys.stdout.flush()

    return crr / tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=0, help="batch size pre gpu")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default='tacred',help='dataset to use')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=3e-5, help='learning rate')
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768,help='hidden size')
    parser.add_argument("--encoder", dest="encoder", type=str,
                        default='bert',help='encoder')
    parser.add_argument("--optim", dest="optim", type=str,
                        default='adamw',help='optimizer')
    
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--ckpt_to_load", dest="ckpt_to_load", type=str,
                        default="None", help="ckpt to load")
    parser.add_argument("--entity_marker", action='store_true', 
                        help="if entity marker or cls")
    parser.add_argument("--train_prop", dest="train_prop", type=float,
                        default=1, help="train set prop")
    
    parser.add_argument("--mode", dest="mode",type=str, 
                        default="CM", help="{CM,OC,CT,OM,OT}")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    args = parser.parse_args()

    # print args
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # Warning
    print("*"*30)
    if args.dataset == 'semeval':
        print("Warning! The results reported on `semeval` may be different from our paper. Because we use the official evaluation script. See `finetune/readme` for more details.")
    print("*"*30)

    # set seed
    set_seed(args)
        
    if not os.path.exists("../log"):
        os.mkdir("../log")
    
    # params for dataloader
    if args.train_prop == 1:
        print("Use all train data!")
        train_set = REDataset("../data/"+args.dataset, "train.txt", args)
    elif args.train_prop == 0.1:
        print("Use 10% train data!")
        train_set = REDataset("../data/"+args.dataset, "train_0.1.txt", args)
    elif args.train_prop == 0.01:
        print("Use 1% train data!")
        train_set = REDataset("../data/"+args.dataset, "train_0.01.txt", args)
    dev_set = REDataset("../data/"+args.dataset, "dev.txt", args)
    test_set = REDataset("../data/"+args.dataset, "test.txt", args)

    train_dataloader = data.DataLoader(train_set, batch_size=args.batch_size_per_gpu, shuffle=True)        
    dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size_per_gpu, shuffle=False)
    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size_per_gpu, shuffle=False)
    
    rel2id = json.load(open(os.path.join("../data/"+args.dataset, "rel2id.json")))
    args.rel_num = len(rel2id)
    
    model = REModel(args)
    model.cuda()
    train(args, model, train_dataloader, dev_dataloader, test_dataloader)
    
    


