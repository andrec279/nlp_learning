"""
TODO: Your code below!

This file should implement all steps described in Part 2, and can be structured however you want.

Rather than using normal BERT, you should use distilbert-base-uncased. This will train faster.

We recommend training on a GPU, either by using HPC or running the command line commands on Colab.

Hints:
    * It will probably be helpful to save intermediate outputs (preprocessed data).
    * To save your finetuned models, you can use torch.save().
"""

from argparse import ArgumentParser
from collections import defaultdict
from datasets import load_dataset

from src.dependency_parse import DependencyParse
# from bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics
import numpy as np
from tqdm import tqdm
import pandas as pd


def data_preproc(split: str):
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    
    ud_data = load_dataset('universal_dependencies', 'en_gum', split=split)
    rel_pos_vocab = set('unk')
    
    pos_labels_df = pd.DataFrame()
    for i, sent in enumerate(ud_data):
        dp = DependencyParse.from_huggingface_dict(sent)
        rel_pos = [(int(dp.heads[i])-1) - i if dp.heads[i] != '0' else 0 for i in range(len(dp.heads))]
        
        row = [{'text': dp.text, 'rel_pos': rel_pos, 'dep_label': dp.deprel}]
        pos_labels_df = pd.concat([pos_labels_df, pd.DataFrame(row)])

        rel_pos_vocab.update(rel_pos)

    pos_labels_df.head(10).to_csv('en_gum_10.tsv', sep="\t", index=False)

    return pos_labels_df['rel_pos'].values.tolist(), pos_labels_df['dep_label'].values.tolist()




def parse_args():
    arg_parser = ArgumentParser()
    #arg_parser.add_argument("method", choices=["spacy", "bert"])
    #arg_parser.add_argument("--data_subset", type=str, default="en_gum")
    arg_parser.add_argument("--split", default="train")
    return arg_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rel_pos, dep_labels = data_preproc(split=args.split)
    print(rel_pos[:10])
    print(dep_labels[:10])