from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse


class SpacyParser(Parser):

    def __init__(self, model_name: str):
        # TODO: Your code here!
        pass

    def parse(sentence: str, tokens: list) -> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        # Your code here!
        raise NotImplementedError
