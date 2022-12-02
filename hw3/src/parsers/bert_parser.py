import json

from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
from src.bert_parser_model import BertParserModel

import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence


class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """Load your saved finetuned model using torch.load().

        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        self.idx_mappers = json.load((open('idx_mappers.json', 'r')))
        self.idx_to_rel_pos = {int(k): v for k, v in self.idx_mappers['idx_to_rel_pos'].items()}
        self.idx_to_deprel = {int(k): v for k, v in self.idx_mappers['idx_to_deprel'].items()}
        self.rel_pos_to_idx = {k: int(v) for k, v in self.idx_mappers['rel_pos_to_idx'].items()}
        self.deprel_to_idx = {k: int(v) for k, v in self.idx_mappers['deprel_to_idx'].items()}


        # TODO: Load your neural net.
        model_params = json.load(open('bert_parser_params.json', 'r'))
        self.model = BertParserModel(model_params)
        self.model.load_state_dict(torch.load(model_path))

    
    def generate_mask(self, sentence, tokenizer):
        '''
        Key function for dealing with differences between BERT Tokenizer tokens and UD Tokens. Creates
        a binary mask set to 0 for any positions in BERT-Tokenized tokens that equal "##" or "-" 
        or immediately follow a "-".
        '''
        word_tokens = tokenizer.tokenize(sentence)
        mask = torch.ones(len(word_tokens), dtype=torch.bool)
        
        for i, token in enumerate(word_tokens):
            if '##' in token: mask[i] = 0
            elif token == '-': mask[i:i+2] = 0
        
        return mask

    def argmax_decode(self, logits, token_mask, tokens, pred_type):
        preds = torch.zeros(len(tokens))
        
        preds_ = logits.argmax(1).squeeze()[1:-1] # remove eos and bos positions
        preds_ = preds_[token_mask][:len(tokens)]

        preds[:preds_.size(0)] = preds_

        min_valid_idx = int(self.idx_mappers['rel_pos_to_idx']['unk']) + 1 
        
        
        # sometimes after removing tokens to align our BERT tokenized sentence with UD tokens, we have less total tokens than the UD
        # dataset. In this case, we pad the rest of the total tokens with 0's, but we need to account for this by mapping those pads
        # to a relative position / dependency relation that doesn't exist in our vocab
        pad = 999 if pred_type == 'rel_pos' else 'unk'
        preds = [self.idx_mappers[f'idx_to_{pred_type}'][str(int(pred.item()))] if pred.item() >= min_valid_idx else pad for pred in preds]

        return preds
    
    def mst_decode(self, logits, token_mask, tokens):
        valid_rel_pos = [int(self.idx_to_rel_pos[idx]) for idx in self.idx_to_rel_pos.keys() if int(idx) > 3]
        G = nx.DiGraph()

        logits = logits[:,:,1:-1]
        for i in range(logits.size(-1)):
            rel_pos_scores = logits[:,:,i].squeeze()

            possible_rel_pos = [pos for pos in valid_rel_pos if 0 <= i + pos < logits.size(-1)]
            possible_rel_pos_ids = [self.rel_pos_to_idx[str(pos)] for pos in possible_rel_pos]
            possible_rel_pos_scores = F.log_softmax(rel_pos_scores[torch.LongTensor(possible_rel_pos_ids)], dim=0)

            rel_pos_dict = dict(zip(possible_rel_pos_ids, possible_rel_pos))

            G.add_node(i)
            for j, pos_id in enumerate(rel_pos_dict):
                G.add_edge(i, i + rel_pos_dict[pos_id], weight=possible_rel_pos_scores[j])
            
        mst = maximum_spanning_arborescence(G)

        heads = torch.IntTensor([999]*len(tokens))
        heads_ = torch.zeros(len(mst.nodes), dtype=torch.int)

        for edge in mst.edges:
            token_idx = edge[1]
            heads_[token_idx] = edge[0]+1
        
        heads_ = heads_[token_mask][:len(tokens)]
        heads[:heads_.size(0)] = heads_

        return heads.detach().cpu().tolist()


    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Build a DependencyParse from the output of your loaded finetuned model.

        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # TODO: Your code here!
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        token_ids_padded = tokenizer(sentence, return_tensors='pt', padding=False)
        token_mask = self.generate_mask(sentence, tokenizer)

        data_dict = {
            'text': sentence,
            'tokens': tokens,
            'head': [],
            'deprel': []   
        }

        rel_pos_logits, deprel_logits = self.model(token_ids_padded)
        data_dict['deprel'] = self.argmax_decode(deprel_logits, token_mask, tokens, 'deprel')

        if self.mst == False:
            rel_pos_pred_values = self.argmax_decode(rel_pos_logits, token_mask, tokens, 'rel_pos')
            for i in range(len(rel_pos_pred_values)):
                if rel_pos_pred_values[i] == 0: head_i = '0'
                else: head_i = str(i + int(rel_pos_pred_values[i]) + 1)
                data_dict['head'].append(head_i)
        
        else:
            rel_pos_pred_values = self.mst_decode(rel_pos_logits, token_mask, tokens)
            data_dict['head'] = [str(i) for i in rel_pos_pred_values]


        return DependencyParse.from_huggingface_dict(data_dict)
    
