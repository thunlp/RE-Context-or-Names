import os 
import pdb 
import torch
import torch.nn as nn 
from pytorch_metric_learning.losses import NTXentLoss
from transformers import BertForMaskedLM, BertForPreTraining, BertTokenizer

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    
    Args:
        inputs: Inputs to mask. (batch_size, max_length) 
        tokenizer: Tokenizer.
        not_mask_pos: Using to forbid masking entity mentions. 1 for not mask.
    
    Returns:
        inputs: Masked inputs.
        labels: Masked language model labels.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

class CP(nn.Module):
    """Contrastive Pre-training model.

    This class implements `CP` model based on model `BertForMaskedLM`. And we 
    use NTXentLoss as contrastive loss function.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        ntxloss: Contrastive loss function.
        args: Args from command line. 
    """
    def __init__(self, args):
        super(CP, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ntxloss = NTXentLoss(temperature=args.temperature)
        self.args = args 
    
    def forward(self, input, mask, label, h_pos, t_pos):
        # masked language model loss
        input = input.view(-1, self.args.max_length)
        mask = mask.view(-1, self.args.max_length)
        label = label.view(-1) # (batch_size * 2)
        h_pos = h_pos.view(-1)
        t_pos = t_pos.view(-1)

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        indice = torch.arange(0, input.size()[0])
        not_mask_pos = torch.zeros((input.size()[0], input.size()[1]), dtype=int)
        not_mask_pos[indice, h_pos] = 1
        not_mask_pos[indice, t_pos] = 1

        m_input, m_labels = mask_tokens(input.cpu(), self.tokenizer, not_mask_pos)
        m_outputs = self.model(input_ids=m_input, labels=m_labels, attention_mask=mask)
        m_loss = m_outputs[1]

        outputs = m_outputs

        # entity marker starter
        batch_size = input.size()[0]
        indice = torch.arange(0, batch_size)
        h_state = outputs[0][indice, h_pos] # (batch_size * 2, hidden_size)
        t_state = outputs[0][indice, t_pos]
        state = torch.cat((h_state, t_state), 1)

        r_loss = self.ntxloss(state, label)

        return m_loss, r_loss



class MTB(nn.Module):
    """Matching the Blanks.

    This class implements `MTB` model based on model `BertForMaskedLM`.

    Attributes:
        model: Model to train.
        tokenizer: Tokenizer.
        bceloss: Binary Cross Entropy loss.
    """
    def __init__(self, args):
        super(MTB, self).__init__()
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bceloss = nn.BCEWithLogitsLoss()
        self.args = args
    

    def forward(self, l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label):
        # compute not mask entity marker
        indice = torch.arange(0, l_input.size()[0])
        l_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 
        r_not_mask_pos = torch.zeros((l_input.size()[0], l_input.size()[1]), dtype=int) 

        # Ensure that `mask_tokens` function doesn't mask entity mention.
        l_not_mask_pos[indice, l_ph] = 1
        l_not_mask_pos[indice, l_pt] = 1

        r_not_mask_pos[indice, r_ph] = 1
        r_not_mask_pos[indice, r_pt] = 1

        # masked language model loss
        m_l_input, m_l_labels = mask_tokens(l_input.cpu(), self.tokenizer, l_not_mask_pos)
        m_r_input, m_r_labels = mask_tokens(r_input.cpu(), self.tokenizer, r_not_mask_pos) 
        m_l_outputs = self.model(input_ids=m_l_input, labels=m_l_labels, attention_mask=l_mask)
        m_r_outputs = self.model(input_ids=m_r_input, labels=m_r_labels, attention_mask=r_mask)
        m_loss = m_l_outputs[1] + m_r_outputs[1]

        # sentence pair relation loss 
        l_outputs = m_l_outputs
        r_outputs = m_r_outputs

        batch_size = l_input.size()[0]
        indice = torch.arange(0, batch_size)
        
        # left output
        l_h_state = l_outputs[0][indice, l_ph] # (batch, hidden_size)
        l_t_state = l_outputs[0][indice, l_pt] # (batch, hidden_size)
        l_state = torch.cat((l_h_state, l_t_state), 1) # (batch, 2 * hidden_size)
        
        # right output 
        r_h_state = r_outputs[0][indice, r_ph] 
        r_t_state = r_outputs[0][indice, r_pt]
        r_state = torch.cat((r_h_state, r_t_state), 1)

        # cal similarity
        similarity = torch.sum(l_state * r_state, 1) # (batch)

        # cal loss
        r_loss = self.bceloss(similarity, label.float())

        return m_loss, r_loss 
        


