from argparse import ArgumentParser
from collections import defaultdict

from src.dependency_parse import DependencyParse
# from bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics


def get_parses(subset: str, test: bool = False) -> List[DependencyParse]:
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    raise NotImplementedError


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("method", choices=["spacy", "bert"])
    arg_parser.add_argument("--data_subset", type=str, default="zh_gsdsimp")
    arg_parser.add_argument("--test", action="store_true")
    
    # SpaCy parser arguments.
    arg_parser.add_argument("--model_name", type=str, default="zh_core_web_sm")

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
    for gold in get_parses(args.data_subset, test=args.test):
        pred = parser.parse(gold.text, gold.tokens)
        for metric, value in get_metrics(pred, gold).items():
            cum_metrics[metric].append(value)
    
    print({metric: np.mean(data) for metric, data in cum_metrics.items()})
