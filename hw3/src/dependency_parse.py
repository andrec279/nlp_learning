"""Class to represent a dependency parse."""

from typing import NamedTuple, List


class DependencyParse(NamedTuple):

    text: str
    tokens: List[str]
    heads: List[str]
    deprel: List[str]

    @classmethod
    def from_huggingface_dict(cls, data_dict):
        return cls(data_dict["text"], data_dict["tokens"], data_dict["head"], data_dict["deprel"])
