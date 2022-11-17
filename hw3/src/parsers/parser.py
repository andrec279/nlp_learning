from src.dependency_parse import DependencyParse

from abc import ABCMeta, abstractmethod


class Parser(metaclass=ABCMeta):

    """Abstract base class for a parser. You should NOT modify this."""

    @abstractmethod
    def parse(sentence: str, tokens: list) -> DependencyParse:
        raise NotImplementedError
