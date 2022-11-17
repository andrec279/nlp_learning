from src.dependency_parse import DependencyParse


def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    # TODO: Your code here!
    return {
        "uas": 0.,
        "las": 0.,
    }
