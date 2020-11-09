import os 
import re
import pdb
import ast 
import json
import random
import argparse
import numpy as np
import pandas as pd 
from tqdm import trange
from transformers import BertTokenizer
from collections import defaultdict, Counter

class EntityMarker():
    """Converts raw text to BERT-input ids and finds entity position.

    Attributes:
        tokenizer: Bert-base tokenizer.
        h_pattern: A regular expression pattern -- * h *. Using to replace head entity mention.
        t_pattern: A regular expression pattern -- ^ t ^. Using to replace tail entity mention.
        err: Records the number of sentences where we can't find head/tail entity normally.
        args: Args from command line. 
    """
    def __init__(self, args=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args

    def tokenize(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False):
        """Tokenizer for `CM`(typically), `CT`, `OC` settings.

        This function converts raw text to BERT-input ids and uses entity-marker to highlight entity 
        position and randomly raplaces entity metion with special `BLANK` symbol. Entity mention can 
        be entity type(If h_type and t_type are't none). And this function returns ids that can be 
        the inputs to BERT directly and entity postion.
        
        Args:
            raw_text: A python list of tokens.
            h_pos_li: A python list of head entity postion. For example, h_pos_li maybe [2, 6] which indicates
                that heah entity mention = raw_text[2:6].
            t_pos_li: The same as h_pos_li.
            h_type: Head entity type. This argument is used when we use `CT` settings as described in paper.
            t_type: Tail entity type.
            h_blank: Whether convert head entity mention to `BLANK`.
            t_blank: Whether convert tail entity mention to `BLANK`.
        
        Returns:
            tokenized_input: Input ids that can be the input to BERT directly.
            h_pos: Head entity position(head entity marker start positon).
            t_pos: Tail entity position(tail entity marker start positon).
        
        Example:
            raw_text: ["Bill", "Gates", "founded", "Microsoft", "."]
            h_pos_li: [0, 2]
            t_pos_li: [3, 4]
            h_type: None
            t_type: None
            h_blank: True
            t_blank: False

            Firstly, replace entity mention with special pattern:
            "* h * founded ^ t ^ ."

            Then, replace pattern:
            "[CLS] [unused0] [unused4] [unused1] founded [unused2] microsoft [unused3] . [SEP]"

            Finally, find the postions of entities and convert tokenized sentence to ids:
            [101, 1, 5, 2, 2631, 3, 7513, 4, 1012, 102]
            h_pos: 1
            t_pos: 5
        """
        tokens = []
        h_mention = []
        t_mention = []
        for i, token in enumerate(raw_text):
            token = token.lower()    
            if i >= h_pos_li[0] and i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_li[0] and i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        # tokenize
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_head = self.tokenizer.tokenize(h_mention)
        tokenized_tail = self.tokenizer.tokenize(t_mention)

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        # If head entity type and tail entity type are't None, 
        # we use `CT` settings to tokenize raw text, i.e. replacing 
        # entity mention with entity type.
        if h_type != None and t_type != None:
            p_head = h_type
            p_tail = t_type

        if h_blank:
            p_text = self.h_pattern.sub("[unused0] [unused4] [unused1]", p_text)
        else:
            p_text = self.h_pattern.sub("[unused0] "+p_head+" [unused1]", p_text)
        if t_blank:
            p_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", p_text)
        else:
            p_text = self.t_pattern.sub("[unused2] "+p_tail+" [unused3]", p_text)
    
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        # If h_pos_li and t_pos_li overlap, we can't find head entity or tail entity.
        try:
            h_pos = f_text.index("[unused0]")
        except:
            self.err += 1
            h_pos = 0
        try:
            t_pos = f_text.index("[unused2]") 
        except:
            self.err += 1
            t_pos = 0

        tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        
        return tokenized_input, h_pos, t_pos
 
    def tokenize_OMOT(self, tokenized_head, tokenized_tail, h_first):
        """Tokenizer for `CM` and `CT` settings.

        This function converts head entity and tail entity to ids.

        Args:
            tokenized_head: Head entity(mention or type in list). Please ensure that this argument has
                been tokenized using bert tokenizer.
            tokenized_tail: Tail entity(mention or type in list). Please ensure that this argument has
                been tokenized using bert tokenizer.
            h_first: Whether head entity is the first entity(i.e. head entity in 
            original sentence is in front of tail entity),
        
        Returns:
            tokenized_input: Input ids that can be the input to BERT directly.
            h_pos: Head entity position(head entity marker start positon).
            t_pos: Tail entity position(tail entity marker start positon).
        """
        tokens = ['[CLS]',]
        if h_first:
            h_pos = 1
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]
            t_pos = len(tokens)
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]
            
        else:
            t_pos = 1
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]
            h_pos = len(tokens)
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]

        tokens.append('[SEP]')
        tokenized_input = self.tokenizer.convert_tokens_to_ids(tokens)

        return tokenized_input, h_pos, t_pos

def sample_trainset(dataset, prop):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)
    
    little_data = []
    reduced_times = 1 / prop
    rel2ins = defaultdict(list)
    for ins in data:
        rel2ins[ins['relation']].append(ins)
    for key in rel2ins.keys():
        sens = rel2ins[key]
        random.shuffle(sens)
        number = int(len(sens) // reduced_times) if len(sens) % reduced_times == 0 else int(len(sens) // reduced_times) + 1
        little_data.extend(sens[:number])
    print("We sample %d instances in " % len(little_data) + dataset +" train set." )
    
    f = open(dataset+"/train_" + str(prop) + ".txt",'w')
    for ins in little_data:
        text = json.dumps(ins)
        f.write(text + '\n')
    f.close()

def get_type2id(dataset):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)

    # Check if entities in data have type.
    if 'type' not in data[0]['h']:
        raise Exception("There is no type infomation is this " + dataset + ".")

    type2id = {'UNK':0}
    for ins in data:
        if 'subj_'+ins['h']['type'] not in type2id:
            type2id['subj_'+ins['h']['type']] = len(type2id)
            type2id['obj_'+ins['h']['type']] = len(type2id)
        if 'subj_'+ins['t']['type'] not in type2id:
            type2id['subj_'+ins['t']['type']] = len(type2id)
            type2id['obj_'+ins['t']['type']] = len(type2id)

    json.dump(type2id, open(dataset+"/type2id.json", 'w'))
    print("File `type2id.json` has been stored in "+dataset+".")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="dataset")
    parser.add_argument("--type2id", action="store_true", help="Whether generating type2id.json or not")
    args = parser.parse_args()

    sample_trainset(args.dataset, 0.01)
    sample_trainset(args.dataset, 0.1)
    # Please make sure entities in `args.dataset` have type infomation. 
    if args.type2id:
        get_type2id(args.dataset)


    
