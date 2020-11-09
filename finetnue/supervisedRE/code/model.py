import torch
import pdb 
import torch.nn as nn
from transformers import BertModel


class REModel(nn.Module):
    """relation extraction model
    """
    def __init__(self, args, weight=None):
        super(REModel, self).__init__()
        self.args = args 
        self.training = True
        
        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        scale = 2 if args.entity_marker else 1
        self.rel_fc = nn.Linear(args.hidden_size*scale, args.rel_num)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if args.ckpt_to_load != "None":
            print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
            ckpt = torch.load("../../../pretrain/ckpt/"+args.ckpt_to_load)
            self.bert.load_state_dict(ckpt["bert-base"])
        else:
            print("*******No ckpt to load, Let's use bert base!*******")
        
    def forward(self, input_ids, mask, h_pos, t_pos, label):
        # bert encode
        outputs = self.bert(input_ids, mask)

        # entity marker
        if self.args.entity_marker:
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1) #(batch_size, hidden_size*2)
        else:
            #[CLS]
            state = outputs[0][:, 0, :] #(batch_size, hidden_size)

        # linear map
        logits = self.rel_fc(state) #(batch_size, rel_num)
        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output    
        




