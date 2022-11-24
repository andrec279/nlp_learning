from argparse import ArgumentParser
from collections import defaultdict
from datasets import load_dataset

from src.dependency_parse import DependencyParse
# from bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics
import numpy as np
from tqdm import tqdm



def get_parses(subset: str, test: bool = False):
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    
    split = 'test' if test==True else 'validation'
    ud_data = load_dataset('universal_dependencies', subset, split=split)
    dependency_parses = []
    for sent in ud_data:
        dependency_parses.append(DependencyParse.from_huggingface_dict(sent))

    return dependency_parses


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("method", choices=["spacy", "bert"])
    arg_parser.add_argument("--data_subset", type=str, default="en_gum")
    arg_parser.add_argument("--test", action="store_true")
    
    # SpaCy parser arguments.
    arg_parser.add_argument("--model_name", type=str, default="en_core_web_sm")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method == "spacy":
        parser = SpacyParser(args.model_name)
    elif args.method == "bert":
        # parser = BertParser()
        pass
    else:
        raise ValueError("Unknown parser")

    cum_metrics = defaultdict(list)
    for gold in tqdm(get_parses(args.data_subset, test=args.test), desc='evaluating dependency parses'):
        pred = parser.parse(gold.text, gold.tokens)
        for metric, value in get_metrics(pred, gold).items():
            cum_metrics[metric].append(value)
    
    print({metric: np.mean(data) for metric, data in cum_metrics.items()})
