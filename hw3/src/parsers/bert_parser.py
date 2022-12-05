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
        '''
        Due to lower than expected UAS / LAS performance with MST decode, I've explained my steps below in hopes of showing
        my implementation of MST decoding is correct.
        '''
        
        # initialize dict to get logit positions from relative positions
        rel_pos_to_idx = self.idx_mappers['rel_pos_to_idx']
        
        # initialize directed graph
        G = nx.DiGraph()
        
        # remove cls and sep tokens from logits
        logits = logits[:,:,1:-1]

        # initialize list of sentence positions and add nodes to G for each position
        sentence_positions = list(range(logits.size(-1)))
        for i in sentence_positions:
            G.add_node(i)
        
        # loop over sentence positions and add edges for every other position in the sentence
        for i in sentence_positions:
            # Get log probabilities over head positions for the given word
            rel_pos_scores = F.log_softmax(logits[:,:,i].squeeze(), dim=0)

            # Get relative positions for all other position nodes in the sentence for a given position
            potential_head_positions = sentence_positions[:i] + sentence_positions[i+1:]
            potential_rel_positions = [pos-i for pos in potential_head_positions]

            # Convert relative position values to class indices for getting log probability scores
            rel_pos_idx = [rel_pos_to_idx[str(pos)] if str(pos) in list(rel_pos_to_idx.keys()) else rel_pos_to_idx['unk'] 
                            for pos in potential_rel_positions]

            # Add an edge for each possible relative position along with log probability weight
            for j in range(len(rel_pos_idx)):
                G.add_edge(potential_head_positions[j], i, weight=rel_pos_scores[rel_pos_idx[j]])  
            
        mst = maximum_spanning_arborescence(G)

        # Initialize "heads" with unk class indices - we will fill this up to len(heads_) after removing undesired BERT tokens
        # (e.g. "##ally") from heads_. Doing this way addresses rare edge cases where len(heads_) < len(tokens) after filtering
        # out undesired tokens
        heads = torch.IntTensor([1]*len(tokens))

        # We initialize heads_ with zeros and fill with values from mst.edges later
        heads_ = torch.zeros(len(mst.nodes), dtype=torch.int)

        # For each "edge" tuple, edge[1] is the token idx and edge[0] is the head token idx identified by MST. Thus we fill 
        # heads_ at position edge[1] with edge[0] + 1, adding 1 to the head token idx to match our token indices with UD.
        # Note that for the root token, there won't be an edge containing the root token's position in edge[1], since that token
        # shouldn't have a head, so the position in heads_ corresponding to the root token will be left as 0, which is what we want.
        for edge in mst.edges:
            token_idx = edge[1]
            heads_[token_idx] = edge[0]+1
        
        # Use token_mask to filter out undesired BERT tokens and truncate to len(tokens)
        heads_ = heads_[token_mask][:len(tokens)]

        # To deal with potential edge cases where heads_ ends up shorter than tokens after mask filtering
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

            # convert all head values to str for proper evaluation
            data_dict['head'] = [str(i) for i in rel_pos_pred_values]

        return DependencyParse.from_huggingface_dict(data_dict)
    
