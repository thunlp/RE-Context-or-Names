import json 
import random
import os 
import sys 
import pdb 
import re 
import torch
import argparse
import numpy as np 
from tqdm import trange
from collections import Counter, defaultdict


def filter_sentence(sentence):
    """Filter sentence.
    
    Filter sentence:
        - head mention equals tail mention
        - head mentioin and tail mention overlap

    Args:
        sentence: A python dict.
            sentence example:
            {
                'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                'r': 'P1'
            }

    Returns:
        True or False. If the sentence contains abnormal conditions 
        above, return True. Else return False

    Raises:
        If sentence's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    head_pos = sentence["h"]["pos"][0]
    tail_pos = sentence["t"]["pos"][0]
    
    if sentence["h"]["name"] == sentence["t"]["name"]:  # head mention equals tail mention
        return True

    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]: # head mentioin and tail mention overlap
        return True
    
    if tail_pos[0] >= head_pos[0] and tail_pos[0] <= head_pos[-1]: # head mentioin and tail mention overlap
        return True  

    return False


def process_data_for_CP(data):
    """Process data for CP. 

    This function will filter NA relation, abnormal sentences,
    and relation of which sentence number is less than 2(This relation
    can't form positive sentence pair).

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        But this function will save two json-formatted files:
            - list_data: A list of sentences.
            - rel2scope: A python dict whose key is relation and value is 
                a scope which is left-closed-right-open `[)`. All sentences 
                in a same scope share the same relation.
            
            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                
                - rel2scope:
                    {
                        'P10': [0, 233],
                        'P1212': [233, 1000],
                        ....
                    }
        
    Raises:
        If data's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    washed_data = {}
    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            if filter_sentence(sen):
                continue
            rel_sentence_list.append(sen)
        if len(rel_sentence_list) < 2:
            continue        
        washed_data[key] = rel_sentence_list

    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
    
    if not os.path.exists("../data/CP"):
        os.mkdir("../data/CP")
    json.dump(list_data, open("../data/CP/cpdata.json","w"))
    json.dump(rel2scope, open("../data/CP/rel2scope.json", 'w'))


def process_data_for_MTB(data):
    """Process data for MTB. 

    This function will filter abnormal sentences, and entity pair of which 
    sentence number is less than 2(This entity pair can't form positive sentence pair).

    Args:
        data: Original data for pre-training and is a dict whose key is relation.
            data example:
                {
                    'P1': [
                        {
                            'token': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }

    Returns: 
        No returns. 
        But this function will save three json-formatted files:
            - list_data: A list of sentences.
            - entpair2scope: A python dict whose key is `head_id#tail_id` and value is 
                a scope which is left-closed-right-open `[)`. All sentences in one same 
                scope share the same entity pair
            - entpair2negpair: A python dict whose key is `head_id#tail_id`. And the value
                is the same format as key, but head_id or tail_id is different(only one id is 
                different). 

            example:
                - list_data:
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                - entpair2scope:
                    {
                        'Q1234#Q2356': [0, 233],
                        'Q135656#Q10': [233, 1000],
                        ....
                    }
                - entpair2negpair:
                    {
                        'Q1234#Q2356': ['Q1234#Q3560','Q923#Q2356', 'Q1234#Q100'],
                        'Q135656#Q10': ['Q135656#Q9', 'Q135656#Q10010', 'Q2666#Q10']
                    }
        
    Raises:
        If data's format isn't the same as described above, 
        this function may raise `key not found` error by Python Interpreter.
    """
    # Maximum number of sentences sharing the same entity pair.
    # This parameter is set for limit the bias towards popular 
    # entity pairs which have many sentences. Of cource, you can
    # change this parameter, but in our expriment, we use 8.
    max_num = 8 

    # We change the original data's format. The ent_data is 
    # a python dict of which key is `head_id#tail_id` and value
    # is sentences which hold this same entity pair.
    ent_data = defaultdict(list)
    for key in data.keys():
        for sentence in data[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            ent_data[head + "#" + tail].append(sentence)

    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        if len(ent_data[key]) < 2:
            continue
        list_data.extend(ent_data[key][0:max_num])
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)

    # We will pre-generate `hard` nagative samples. The entpair2negpair
    # is a python dict of which key is `head_id#tail_id`. And the value of the dict
    # is the same format as key, but head_id or tail_id is different(only one id is 
    # different). 
    entpair2negpair = defaultdict(list)
    entpairs = list(entpair2scope.keys())
    entpairs.sort(key=lambda a: a.split("#")[0])
    for i in range(len(entpairs)):
        head = entpairs[i].split("#")[0]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[0] != head:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    entpairs.sort(key=lambda a: a.split("#")[1])
    for i in range(len(entpairs)):
        tail = entpairs[i].split("#")[1]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[1] != tail:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])

    if not os.path.exists("../data/MTB"):
        os.mkdir("../data/MTB")
    json.dump(entpair2negpair, open("../data/MTB/entpair2negpair.json","w"))
    json.dump(entpair2scope, open("../data/MTB/entpair2scope.json", "w"))
    json.dump(list_data, open("../data/MTB/mtbdata.json", "w"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="MTB", help="{MTB,CP}")
    args = parser.parse_args()
    set_seed(42)

    data = json.load(open("../data/exclude_fewrel_distant.json"))
    if args.dataset == "CP":
        process_data_for_CP(data)

    elif args.dataset == "MTB":
        process_data_for_MTB(data)
    
    























