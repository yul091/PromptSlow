import numpy as np
from typing import Optional, List, Union
import torch
from transformers import (
    BertTokenizerFast, 
    BertForMaskedLM,
    BartForConditionalGeneration,
)
import nltk
from nltk.corpus import wordnet as wn
from .base import SlowAttacker
from OpenAttack.text_process.tokenizer import Tokenizer, PunctTokenizer
from OpenAttack.attack_assist.substitute.word import WordNetSubstitute
from utils import GrammarChecker, ENGLISH_FILTER_WORDS, DEFAULT_TEMPLATES
        
        
class PromptAttacker(SlowAttacker):
    def __init__(
        self, 
        device: torch.device = None,
        tokenizer: Union[Tokenizer, BertTokenizerFast] = None,
        model: Union[BertForMaskedLM, BartForConditionalGeneration] = None,
        max_len: int = 64,
        max_per: int = 3,
        task: str = 'seq2seq',
        fitness: str = 'adaptive',
        select_beams: int = 1,
        eos_weight: float = 0.5,
        cls_weight: float = 0.5,
        use_combined_loss: bool = False,
    ):
        super(PromptAttacker, self).__init__(
            device, tokenizer, model, max_len, max_per, task, fitness,
            select_beams, eos_weight, cls_weight, use_combined_loss,
        )
        self.beam_size = 50
        self.unk_token = tokenizer.unk_token
        self.filter_words = set(ENGLISH_FILTER_WORDS)
        self.triggers = ["the" for _ in range(max_per)]
        self.substitute = WordNetSubstitute()
        self.default_tokenizer = PunctTokenizer()

    def compute_loss(self, text: list, labels: list):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        cls_loss = self.get_cls_loss(text, labels)
        return (loss_list, cls_loss)
    
    def mutation(
        self, 
        context: str, 
        sentence: str, 
        grad: torch.Tensor,
        label: str, 
        modify_pos: List[int],
    ):
        new_strings = []
        trigger_len = len(self.triggers)
        modified_pos = set(modify_pos)
        orig_sent = self.default_tokenizer.tokenize(sentence, pos_tagging=False)
        if modify_pos:
            orig_sent = orig_sent[trigger_len:]
        
        def removeBPE(word: str):
            if word.startswith('▁'):
                return word.lstrip('▁').lower()
            if word.startswith('Ġ'):
                return word.lstrip('Ġ').lower()
            return word.lower()
        
        important_tensor = (-grad.sum(1)).argsort()[:self.beam_size] # sort token ids w.r.t. gradient
        important_tokens = self.tokenizer.convert_ids_to_tokens(important_tensor.tolist())
        
        for i in range(trigger_len):
            if i in modified_pos:
                continue
            for cw in important_tokens:
                cw = removeBPE(cw)
                tt = self.triggers[:i] + [cw] + self.triggers[i + 1:]
                xt = self.default_tokenizer.detokenize(tt + orig_sent)
                new_strings.append((i, xt))
        
        return new_strings