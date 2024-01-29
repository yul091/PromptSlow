# coding:utf8
import random
from collections import namedtuple
from copy import deepcopy
from typing import List, Union
import numpy
from transformers import PreTrainedTokenizerFast

random.seed(1234)


Config = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'position_size', 'position_dim',
                     'word_input_size', 'sent_input_size',
                     'word_GRU_hidden_units', 'sent_GRU_hidden_units',
                     'pretrained_embedding', 'word2id', 'id2word',
                     'dropout'])


class Document():
    def __init__(
        self, 
        query: List[str] = None, 
        reference: List[str] = None,
        instruction: str = None,
        demonstrations: List[str] = None,
        prefix: str = None,
    ):
        self.query = query
        self.reference = reference
        self.prefix = prefix
        self.instruction = instruction
        self.demonstrations = demonstrations


# a bunch of converter functions
def tokens_to_sentences(
    token_list: List[Union[str, List[str]]], 
    tokenizer: PreTrainedTokenizerFast,
) -> List[str]:
    # convert a token list to sents list
    # this is a cheap fix, might need better way to do it
    if isinstance(token_list[0], list):
        sents_list = token_list
    else:
        sents_list = []
        counter = 0
        for i, token in enumerate(token_list):
            if token == '.' or token == '!' or token == '?':
                sents_list.append(token_list[counter:i + 1])  # include .!? in sents
                counter = i + 1

    # sents_list = [" ".join(s) for s in sents_list]
    sents_list = [tokenizer.convert_tokens_to_string(s) for s in sents_list]
    sents_list = [s.replace("<s>", '') for s in sents_list]
    sents_list = [s.replace("</s>", '') for s in sents_list]

    return sents_list


def remove_control_tokens(text: Union[str, List[str]]) -> Union[str, List[str]]:
    if type(text) == str:
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
    # list of strings
    if type(text) == list:
        text = [s.replace("<s>", "") for s in text if type(s) == str]
        text = [s.replace("</s>", "") for s in text if type(s) == str]
    return text


def prepare_data(doc: Document, word2id: dict) -> numpy.ndarray:
    data = deepcopy(doc.content)
    max_len = -1  # this is for padding
    for sent in data:
        words = sent.strip().split()
        max_len = max(max_len, len(words))
    sent_list = []

    for sent in data:
        words = sent.lower().strip().split()
        sent = [word2id[word] for word in words]
        if len(sent) == 0:
            continue
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        sent_list.append(sent)

    sent_array = numpy.array(sent_list)
    return sent_array