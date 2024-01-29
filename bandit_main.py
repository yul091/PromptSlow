import sys 
sys.dont_write_bytecode = True
import argparse
import torch
import numpy as np
from typing import List, Dict, Any, Union
from helper import Document, tokens_to_sentences
from Dataset import prompt_dataset, BatchDataLoader
from reinforce import ReinforceReward
from huggingface_api import add_model_args
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class BanditPrompt:
    
    def __init__(self, args: argparse.Namespace, device: str = "cuda"):
        self.args = args
        self.task = args.task
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.my_model_name = args.my_model_name
        self.device = device
        self.max_length = args.max_length
        self.max_new_tokens = args.max_new_tokens
        self.saving_step = args.saving_step
        self.output_dir = args.output_dir
        self.n_train_samples = args.n_train_samples
        self.n_test_samples = args.n_test_samples
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
        self.chunk_size = args.chunk_size
        # bandit hyperparams
        self.rouge_metric = args.rouge_metric
        self.batch_size = args.batch_size
        self.rl_baseline_method = args.rl_baseline_method
        if args.prt_inf and np.random.randint(0, 100) == 0:
            self.prt = True
        else:
            self.prt = False
        self.length_limit = args.length_limit
            
        self.config = AutoConfig.from_pretrained(self.my_model_name, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.my_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.my_model_name, 
            config=self.config,
        ).to(self.device)
        
        # Loss and Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        
    def prepare_inputs(self, inputs: Dict[str, Union[Any, torch.Tensor]]):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
    
    def run(self):
        
        reinforce = ReinforceReward(
            args=self.args, 
            rouge_metric=self.rouge_metric,
            b=self.batch_size, 
            rl_baseline_method=self.rl_baseline_method,
            loss_method=1,
        )
        instruction, demonstrations, prefix, train_dataset, test_dataset = prompt_dataset(self.task)
        # For training w/ RL, we consider only batch_size = 1
        
        print(" ** Start training with RL ** ")
        for step, batch in enumerate(BatchDataLoader(train_dataset, shuffle=True)):
            self.model.train()
            # batch: Dict[str, List[str]]
            doc = Document(
                reference=batch["reference"],
                instruction=instruction,
                demonstrations=demonstrations,
                prefix=prefix,
            )
            
            tokens = self.tokenizer.tokenize(batch['query'][0])
            doc.query = tokens_to_sentences(tokens, self.tokenizer)
            if self.args.oracle_length == -1:  # use true oracle length
                oracle_query_sent_num = len(doc.query)
            else:
                oracle_query_sent_num = self.args.oracle_length
            
            if len(doc.query) == 0:
                continue
            print(doc.query)
            
            inputs = self.tokenizer(
                doc.query,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = self.prepare_inputs(inputs)
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            print(probs)
            
            loss, reward = reinforce.train(
                probs, 
                doc,
                max_num_of_sents=oracle_query_sent_num,
                max_num_of_bytes=self.length_limit,
                prt=self.prt,
            )
            
            break
    
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--my_model_name", type=str, default="bert-base-cased")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    # Training arguments 
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--saving_step", type=int, default=100)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--prt_inf', action='store_true')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')
    parser.add_argument('--oracle_length', type=int, default=-1,
                        help='-1 for giving actual oracle number of sentences, otherwise choose a fixed number of sentences')
    # Data arguments
    parser.add_argument("--task", type=str, default="conversational")
    parser.add_argument("--n_train_samples", type=int, default=-1)
    parser.add_argument("--n_test_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument('--rouge_metric', type=str, default='avg_f')
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2
        
    if args.length_limit > 0:
        args.oracle_length = 2

    # generation_pipeline(args)
    banditprompt = BanditPrompt(args)
    banditprompt.run()