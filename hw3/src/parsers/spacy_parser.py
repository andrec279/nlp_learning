from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
import spacy


class SpacyParser(Parser):

    def __init__(self, model_name: str):
        # TODO: Your code here!
        self.model = spacy.load(model_name)

    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        # Your code here!
        doc = spacy.tokens.doc.Doc(self.model.vocab, words=tokens)
        parse = self.model(doc)
        
        data_dict = {
            'text': sentence,
            'tokens': tokens,
            'head': [],
            'head_text': [],
            'deprel': []   
        }
        for token in parse:
            if token.i == token.head.i: head_i = '0'
            else: head_i = str(token.head.i + 1)
            data_dict['head'].append(head_i)
            data_dict['head_text'].append(token.text)
            data_dict['deprel'].append(token.dep_)
  
        return DependencyParse.from_huggingface_dict(data_dict)
