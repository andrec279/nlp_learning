from src.dependency_parse import DependencyParse
import numpy as np


def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    predicted_heads = np.array(predicted.heads)
    predicted_deprels = np.array(predicted.deprel)
    gold_heads = np.array(labeled.heads)
    gold_deprels = np.array(labeled.deprel)

    return {
        "uas": sum(predicted_heads==gold_heads)/len(gold_heads),
        "las": sum((predicted_heads==gold_heads)&(predicted_deprels==gold_deprels))/len(gold_heads),
    }
