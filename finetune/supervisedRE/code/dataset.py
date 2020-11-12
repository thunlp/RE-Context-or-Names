
import os 
import re
import ast 
import sys 
sys.path.append("..")
import json 
import pdb
import random 
import torch 
import numpy as np 
from torch.utils import data
sys.path.append("../../../")
from utils.utils import EntityMarker


class REDataset(data.Dataset):
    """Data loader for semeval, tacred
    """
    def __init__(self, path, mode, args):
        data = []
        with open(os.path.join(path, mode)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                ins = json.loads(line)
                data.append(ins)
        
            
        entityMarker = EntityMarker(args)
        tot_instance = len(data)

        # load rel2id and type2id
        if os.path.exists(os.path.join(path, "rel2id.json")):
            rel2id = json.load(open(os.path.join(path, "rel2id.json")))
        else:
            raise Exception("Error: There is no `rel2id.json` in "+ path +".")
        if os.path.exists(os.path.join(path, "type2id.json")):
            type2id = json.load(open(os.path.join(path, "type2id.json")))
        else:
            print("Warning: There is no `type2id.json` in "+ path +", If you want to train model using `OT`, `CT` settings, please firstly run `utils.py` to get `type2id.json`.")
    
        print("pre process " + mode)
        # pre process data
        self.input_ids = np.zeros((tot_instance, args.max_length), dtype=int)
        self.mask = np.zeros((tot_instance, args.max_length), dtype=int) 
        self.h_pos = np.zeros((tot_instance), dtype=int)
        self.t_pos = np.zeros((tot_instance), dtype=int)
        self.label = np.zeros((tot_instance), dtype=int)

        for i, ins in enumerate(data):
            self.label[i] = rel2id[ins["relation"]]            
            # tokenize
            if args.mode == "CM":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'])
            elif args.mode == "OC":
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], None, None, True, True)
            elif args.mode == "CT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                ids, ph, pt = entityMarker.tokenize(data[i]["token"], data[i]['h']['pos'], data[i]['t']['pos'], h_type, t_type)
            elif args.mode == "OM":
                head = entityMarker.tokenizer.tokenize(ins['h']['name'])
                tail = entityMarker.tokenizer.tokenize(ins['t']['name'])
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT(head, tail, h_first)
            elif args.mode == "OT":
                h_type = "[unused%d]" % (type2id['subj_'+ins['h']['type']] + 10)
                t_type = "[unused%d]" % (type2id['obj_'+ins['t']['type']] + 10)
                h_first = ins['h']['pos'][0] < ins['t']['pos'][0]
                ids, ph, pt = entityMarker.tokenize_OMOT([h_type,], [t_type,], h_first)
            else:
                raise Exception("No such mode! Please make sure that `mode` takes the value in {CM,OC,CT,OM,OT}")

            length = min(len(ids), args.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, args.max_length-1) 
            self.t_pos[i] = min(pt, args.max_length-1) 
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        mask = self.mask[index]
        h_pos = self.h_pos[index]
        t_pos = self.t_pos[index]
        label = self.label[index]

        return input_ids, mask, h_pos, t_pos, label, index
