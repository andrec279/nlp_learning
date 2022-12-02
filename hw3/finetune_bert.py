from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
import json

from src.dependency_parse import DependencyParse
from src.bert_parser_model import BertParserModel

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
#import matplotlib.pyplot as plt

from transformers import DistilBertTokenizer, DistilBertModel


# Global vars
BATCH_SIZE = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class UDDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data):
        self.text = data['text']
        self.rel_pos = data['rel_pos']
        self.deprel = data['deprel']
        self.ud_tokens = data['ud_tokens']
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        """
        Triggered when you call dataset[i]
        """

        return (self.text[idx], self.rel_pos[idx], self.deprel[idx], self.ud_tokens[idx])


def data_preproc(split: str):
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    
    ud_data = load_dataset('universal_dependencies', 'en_gum', split=split)
    rel_pos_vocab = ['pad', 'bos', 'eos', 'unk']
    deprel_vocab = ['pad', 'bos', 'eos', 'unk']
    
    tokens = []
    pos_labels_df = pd.DataFrame()
    for i, sent in enumerate(ud_data):
        dp = DependencyParse.from_huggingface_dict(sent)
        rel_pos = [(int(dp.heads[i])-1) - i if dp.heads[i] != '0' else 0 for i in range(len(dp.heads))]
        
        row = [{'text': dp.text, 'rel_pos': rel_pos, 'dep_label': dp.deprel}]
        pos_labels_df = pd.concat([pos_labels_df, pd.DataFrame(row)])

        rel_pos_vocab += [pos for pos in list(set(rel_pos)) if pos not in rel_pos_vocab]
        deprel_vocab += [deprel for deprel in list(set(dp.deprel)) if deprel not in deprel_vocab]
        tokens.append(dp.tokens)

    pos_labels_df.head(10).to_csv('en_gum_10.tsv', sep="\t", index=False)

    preproc_data = {'text': pos_labels_df['text'].values.tolist(), 'rel_pos': pos_labels_df['rel_pos'].values.tolist(), 
                    'deprel': pos_labels_df['dep_label'].values.tolist(), 'ud_tokens': tokens}

    idx_mappers = {'idx_to_rel_pos': dict([(i, rel_pos) for i, rel_pos in enumerate(rel_pos_vocab)]), 
                   'rel_pos_to_idx': dict([(rel_pos, i) for i, rel_pos in enumerate(rel_pos_vocab)]), 
                   'idx_to_deprel': dict([(i, deprel) for i, deprel in enumerate(deprel_vocab)]),
                   'deprel_to_idx': dict([(deprel, i) for i, deprel in enumerate(deprel_vocab)])}

    return preproc_data, idx_mappers


def val_to_idx_padded(vals, mapper, max_seq_len):
    # Takes in list of lists of vals, maps them to their idxs in the vocabulary, and applies padding
    output = []
    
    pad_idx, bos_idx, eos_idx = (mapper['pad'], mapper['bos'], mapper['eos'])
    for row in vals:
        padded_idxs = [bos_idx] + [mapper[val] if val in mapper else mapper['unk'] for val in row] + [eos_idx] + [pad_idx]*(max_seq_len-len(row))
        output.append(padded_idxs)
    
    return torch.LongTensor(output)


def generate_masks(text_list, tokens_tensor):
    '''
    Key function for dealing with differences between BERT Tokenizer tokens and UD Tokens. Creates
    a binary mask set to 0 for any positions in BERT-Tokenized tokens that equal "##" or "-" 
    or immediately follow a "-". Note that this mask is intended to be applied to logits after
    model.forward() on padded text token IDs, so the zeroed-out positions are offset by 1 to account 
    for the bos token.
    '''
    mask = torch.ones_like(tokens_tensor)
    for i, text in enumerate(text_list):
        tokens = tokenizer.tokenize(text)
        for j, token in enumerate(tokens):
            if '##' in token: mask[i][j+1] = 0
            elif token == '-': mask[i][j+1:j+3] = 0
    return mask


def build_tensor_dataset(split='train'):
    preproc_data, idx_mappers = data_preproc('train')
    
    return UDDataset(preproc_data), idx_mappers


def batch_collate(batch, idx_mappers):
    text = [b[0] for b in batch]
    rel_pos = [b[1] for b in batch]
    deprel = [b[2] for b in batch]

    token_ids_padded = tokenizer(text, return_tensors='pt', padding=True)

    max_seq_len = max(len(labels) for labels in deprel)
    
    rel_pos_mapper, deprel_mapper = (idx_mappers['rel_pos_to_idx'], idx_mappers['deprel_to_idx'])
    rel_pos_ids_padded = val_to_idx_padded(rel_pos, rel_pos_mapper, max_seq_len)
    deprel_ids_padded = val_to_idx_padded(deprel, deprel_mapper, max_seq_len)
    
    bert_tokens_mask = generate_masks(text, token_ids_padded.input_ids)

    return token_ids_padded, rel_pos_ids_padded, deprel_ids_padded, bert_tokens_mask


def align_tokens(rel_pos_logits_, deprel_logits_, rel_pos_ids, deprel_ids, bert_tokens_mask):
  '''
  Key function for dealing with differences between BERT Tokenizer tokens and UD Tokens. Applies
  the binary mask created in generate_masks to logits, then aligns the dimensions with the label 
  dimensions using pad tokens.
  '''
  
  max_len = rel_pos_ids.size(1)
  rel_pos_logits = torch.zeros((rel_pos_logits_.size(0), rel_pos_logits_.size(1), max_len))
  deprel_logits = torch.zeros((deprel_logits_.size(0), deprel_logits_.size(1), max_len))

  for i in range(len(rel_pos_ids)):
    mask = bert_tokens_mask[i].nonzero().squeeze()
    
    rel_pos_logits_i = rel_pos_logits_[i][:,mask][:,:max_len]
    deprel_logits_i = deprel_logits_[i][:,mask][:,:max_len]
    rel_pos_logits[i][:, :rel_pos_logits_i.size(1)] = rel_pos_logits_i
    deprel_logits[i][:, :deprel_logits_i.size(1)] = deprel_logits_i
  
  return rel_pos_logits, deprel_logits


def finetune_epoch(parser: BertParserModel,
                   train_dataset,
                   batch_collater,
                   batch_size=BATCH_SIZE,
                   lr=1e-4,
                   lamb=0.25,
                   device=device):
  
  parser = parser.to(device)
  parser.train()
  loader = DataLoader(train_dataset, batch_size, collate_fn=batch_collater)

  params_to_update = []
  for name, param in parser.named_parameters():
      if param.requires_grad == True:
          params_to_update.append(param)

  optimizer = torch.optim.Adam(params_to_update, lr=lr, eps=1e-08)
  criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

  idx = 0
  train_loss_curve = []
  for batch in tqdm(loader, desc="Train"):
    token_ids, rel_pos_ids, deprel_ids, bert_tokens_mask = batch

    # Train loop.
    optimizer.zero_grad()
    rel_pos_logits_, deprel_logits_ = parser(token_ids.to(device))
    rel_pos_logits, deprel_logits = align_tokens(rel_pos_logits_, deprel_logits_, rel_pos_ids, deprel_ids, bert_tokens_mask)
  
    loss_rel_pos = criterion(rel_pos_logits, rel_pos_ids)
    loss_deprel = criterion(deprel_logits, deprel_ids)
    loss = lamb*loss_rel_pos + (1-lamb)*loss_deprel
    loss.backward()
    
    # Have gradients at this point.
    nn.utils.clip_grad_norm_(parser.parameters(), max_norm=1.0, norm_type=2)
    optimizer.step()

    train_loss_curve.append(loss.item())
    idx += 1
  
  return train_loss_curve
  

@torch.no_grad()
def evaluate(parser: BertParserModel, eval_dataset, batch_collater):
  parser.eval()
  loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=batch_collater)
  n_rel_pos_correct = 0
  n_deprel_correct = 0
  n_total = 0
  
  idx = 0
  for batch in tqdm(loader, desc="Eval"):
    token_ids, rel_pos_ids, deprel_ids, bert_tokens_mask = batch

    # Compute accuracy.
    rel_pos_logits_, deprel_logits_ = parser(token_ids.to(device))
    rel_pos_logits, deprel_logits = align_tokens(rel_pos_logits_, deprel_logits_, rel_pos_ids, deprel_ids, bert_tokens_mask)
  
    rel_pos_correct = rel_pos_logits.argmax(1) == rel_pos_ids
    deprel_correct = deprel_logits.argmax(1) == deprel_ids
    n_rel_pos_correct += rel_pos_correct.sum().item()
    n_deprel_correct += deprel_correct.sum().item()
    n_total += rel_pos_correct.numel()

    idx += 1

  print("Rel Pos Acc:", n_rel_pos_correct / n_total)
  print("Dep Rel Acc:", n_deprel_correct / n_total)


def finetune(parser: BertParserModel, train_dataset, eval_dataset, batch_collater, lamb, n_epochs: int = 3):
  print("Using device:", device)
  
  losses = []
  for epoch in range(n_epochs):
    print(f"Starting epoch {epoch}...")
    loss_curves = finetune_epoch(parser, train_dataset, batch_collater, lamb=lamb)
    losses += loss_curves
    evaluate(parser, eval_dataset, batch_collater)
  
  print('Final training loss:', losses[-1])
  
#   plt.plot(np.arange(len(losses)), losses)
#   plt.ylabel('Training loss')
#   plt.title(f'Dependency Parser Training Loss Curves @ lambda = {lamb}')
#   plt.show()
  
  torch.save(parser.state_dict(), f'bert-parser-{lamb}.pt')


if __name__ == '__main__':
    ud_dataset_train, idx_mappers = build_tensor_dataset(split='train')
    ud_dataset_eval, _ = build_tensor_dataset(split='val')

    model_params = {'n_rel_pos': len(list(idx_mappers['rel_pos_to_idx'].keys())),
                    'n_deprel': len(list(idx_mappers['deprel_to_idx'].keys())),
                    'd_hidden': 768}
    
    json.dump(model_params, open('bert_parser_params.json', 'w'))
    json.dump(idx_mappers, open('idx_mappers.json', 'w'))

    batch_collater = partial(batch_collate, idx_mappers=idx_mappers)

    for lamb in [0.25, 0.5, 0.75]:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        parser = BertParserModel(model_params, model).to(device)
        finetune(parser, ud_dataset_train, ud_dataset_eval, batch_collater, lamb=lamb)
