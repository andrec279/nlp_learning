import torch.nn as nn
from transformers import DistilBertModel

class BertParserModel(nn.Module):

  def __init__(self, model_params, model=DistilBertModel.from_pretrained("distilbert-base-uncased")):
    super().__init__()
    self.model = model
    self.position_project = nn.Linear(model_params['d_hidden'], model_params['n_rel_pos'])
    self.deprel_project = nn.Linear(model_params['d_hidden'], model_params['n_deprel'])

  def forward(self, tokens): # Takes in batch of tokens [batch_size, padded_seq_len]
    token_hidden_states, = self.model(**tokens).values() # Computes hidden states for tokens -> [batch_size, padded_seq_len, 768]

    position_scores = self.position_project(token_hidden_states).transpose(1, 2) # -> [batch_size, n_positions, padded_seq_len]
    deprel_scores = self.deprel_project(token_hidden_states).transpose(1, 2) # -> [batch_size, n_deprels, padded_seq_len]

    return position_scores, deprel_scores