import os
import sys 
sys.dont_write_bytecode = True
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from typing import List, Dict, Any, Union, Tuple
from helper import Document, tokens_to_sentences
from Dataset import prompt_dataset, BatchDataLoader
from reinforce import ReinforceReward, return_summary_index
from huggingface_api import add_model_args
from rougefonc import from_summary_index_compute_rouge
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
)


np.set_printoptions(precision=4, suppress=True)


def reinforce_loss(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerFast,
    probs: torch.Tensor, 
    doc: Document, 
    max_num_of_sents: int = 3, 
    max_num_of_bytes: int = -1,
    std_rouge: bool = False, 
    rouge_metric: str = "all", 
) -> Tuple[float, float, int, float, float, int]:
    # Sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    max_num_of_sents = min(len(probs_numpy), max_num_of_sents)  # max of sents# in doc and sents# in summary

    rl_baseline_summary_index, _ = return_summary_index(
        probs_numpy, 
        probs,
        sample_method="greedy", 
        max_num_of_sents=max_num_of_sents,
    )
    rl_baseline_summary_index = sorted(rl_baseline_summary_index)
    base_qr, base_lr, base_length = from_summary_index_compute_rouge(
        args=args,
        doc=doc, 
        summary_index=rl_baseline_summary_index, 
        tokenizer=tokenizer,
        std_rouge=std_rouge,
        rouge_metric=rouge_metric,
        max_num_of_bytes=max_num_of_bytes,
        output_length=True,
    )
    lead3_qr, lead3_lr, lead3_length = from_summary_index_compute_rouge(
        args=args,
        doc=doc, 
        summary_index=range(max_num_of_sents), 
        tokenizer=tokenizer,
        std_rouge=std_rouge,
        rouge_metric=rouge_metric,
        max_num_of_bytes=max_num_of_bytes,
        output_length=True,
    )
    return base_qr, base_lr, base_length, lead3_qr, lead3_lr, lead3_length



class BanditPrompt:
    
    def __init__(self, args: argparse.Namespace, device: str = "cuda"):
        self.args = args
        self.task = args.task
        self.lr = args.lr
        self.seed = args.seed
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
        
        # Reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Load datasets
        self.instruction, self.demonstrations, self.prefix, train_dataset, test_dataset = prompt_dataset(self.task)
        if self.n_train_samples > 0:
            n_train_samples = min(self.n_train_samples, len(train_dataset))
            train_dataset = train_dataset.select(range(n_train_samples))
            
        if self.n_test_samples > 0:
            n_test_samples = min(self.n_test_samples, len(test_dataset))
            test_dataset = test_dataset.select(range(n_test_samples))
        
        # For training w/ RL, we consider only batch_size = 1
        self.train_dataloader = BatchDataLoader(train_dataset, shuffle=True)
        self.test_dataloader = BatchDataLoader(test_dataset, shuffle=False)
        
        
    def prepare_inputs(self, inputs: Dict[str, Union[Any, torch.Tensor]]):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    
    def forward_step(
        self, 
        model: torch.nn.Module,
        batch: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[Document, torch.Tensor, int]:
        
        doc = Document(
            reference=batch["reference"],
            instruction=self.instruction,
            demonstrations=self.demonstrations,
            prefix=self.prefix,
        )
        
        tokens = self.tokenizer.tokenize(batch['query'][0])
        doc.query = tokens_to_sentences(tokens, self.tokenizer)
        if self.args.oracle_length == -1:  # use true oracle length
            oracle_query_sent_num = len(doc.query)
        else:
            oracle_query_sent_num = self.args.oracle_length
        
        if len(doc.query) == 0 or len(doc.reference) == 0:
            return doc, None, oracle_query_sent_num
        # print(doc.query)
        
        inputs = self.tokenizer(
            doc.query,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = self.prepare_inputs(inputs)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)
        return doc, probs, oracle_query_sent_num
        
    
    def run(self):
        # init statistics
        quality_reward_list, length_reward_list = [], []
        best_eval_reward = 0.
        
        reinforce = ReinforceReward(
            args=self.args, 
            tokenizer=self.tokenizer,
            rouge_metric=self.rouge_metric,
            b=self.batch_size, 
            rl_baseline_method=self.rl_baseline_method,
            loss_method=1,
        )
        
        print(" ** Start training with RL ** ")
        for epoch in range(self.num_epochs):
            step_in_epoch = 0
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader.dataset))
            for step, batch in pbar:
                self.model.train()
                step_in_epoch += 1
                # batch: Dict[str, List[str]]
                doc, probs, oracle_query_sent_num = self.forward_step(self.model, batch)
                if probs is None:
                    continue
                
                loss, quality_reward, length_reward = reinforce.train(
                    probs, 
                    doc,
                    max_num_of_sents=oracle_query_sent_num,
                    max_num_of_bytes=self.length_limit,
                    prt=self.prt,
                )
                
                if self.prt:
                    print('Probabilities: ', probs.squeeze().data.cpu().numpy())
                    print('-' * 80)
                
                quality_reward_list.append(quality_reward)
                length_reward_list.append(length_reward)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)  # gradient clipping
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pbar.set_description('Epoch %d Step %d Quality reward %.4f Length reward %.4f' % (
                    epoch, step_in_epoch, quality_reward, length_reward))
            
                if (step_in_epoch) % self.saving_step == 0 and step_in_epoch != 0:
                    print('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) + \
                        ' quality reward: ' + str(np.mean(quality_reward_list)) + \
                            ' length reward: ' + str(np.mean(length_reward_list)))
                    quality_reward_list, length_reward_list = [], []

            # if (step_in_epoch) % 10000 == 0 and step_in_epoch != 0:
            print(" ** Evaluation ** ")
            eval_qr, eval_lr, lead3_qr, lead3_lr = self.evaluate(self.model, self.args, self.test_dataloader)
            avg_qr_f1 = np.mean(list(eval_qr.values()))
            avg_lead3_qr_f1 = np.mean(list(lead3_qr.values()))
            if avg_qr_f1 > best_eval_reward:
                best_eval_reward = avg_qr_f1
                print("saving model %s with quality reward %.4f, length reward %.4f, lead quality reward %.4f, lead length reward %.4f" % (
                    self.output_dir, avg_qr_f1, eval_lr, avg_lead3_qr_f1, lead3_lr))
                # torch.save(extract_net, model_name)
                self.model.save_pretrained(self.output_dir)
            print("Epoch %d: eval quality reward %.4f, length reward %.4f, lead quality reward %.4f, lead length reward %.4f" % (
                    epoch, avg_qr_f1, eval_lr, avg_lead3_qr_f1, lead3_lr))


    def evaluate(self, model: torch.nn.Module, args: argparse.Namespace, eval_dataloder: BatchDataLoader):
        model.eval()
        eval_q_rewards, eval_l_rewards, eval_lengths, lead3_q_rewards, lead3_l_rewards, lead3_lengths = [], [], [], [], [], []

        pbar = tqdm(enumerate(eval_dataloder), total=len(eval_dataloder.dataset))
        for step, batch in pbar:
            doc, probs, oracle_query_sent_num = self.forward_step(model, batch)
            if probs is None:
                continue

            compute_score = (step == len(eval_dataloder.dataset) - 1) or (args.std_rouge is False)
            qr, lr, length, lead3_qr, lead3_lr, lead3_length = reinforce_loss(
                args, 
                self.tokenizer,
                probs,
                doc,
                max_num_of_sents=oracle_query_sent_num,
                max_num_of_bytes=args.length_limit,
                std_rouge=args.std_rouge, 
                rouge_metric='all',
            )
            
            if compute_score:
                eval_q_rewards.append(qr) # list of tuples
                eval_l_rewards.append(lr) # list of floats
                eval_lengths.append(length) # list of ints
                lead3_q_rewards.append(lead3_qr) # list of tuples
                lead3_l_rewards.append(lead3_lr) # list of floats
                lead3_lengths.append(lead3_length) # list of ints

        avg_eval_qr = np.mean(eval_q_rewards, axis=0)
        avg_eval_qr = {
            "rouge_1_f": avg_eval_qr[2],
            "rouge_2_f": avg_eval_qr[5],
            "rouge_l_f": avg_eval_qr[8],
        }
        avg_eval_lr = np.mean(eval_l_rewards, axis=0)
        avg_eval_length = np.mean(eval_lengths, axis=0)
        avg_lead3_qr = np.mean(lead3_q_rewards, axis=0)
        avg_lead3_qr = {
            "rouge_1_f": avg_lead3_qr[2],
            "rouge_2_f": avg_lead3_qr[5],
            "rouge_l_f": avg_lead3_qr[8],
        }
        avg_lead3_lr = np.mean(lead3_l_rewards, axis=0)
        avg_lead3_length = np.mean(lead3_lengths, axis=0)
        print('Avg eval quality reward: ', avg_eval_qr)
        print('Avg eval length: %.4f, reward: %.4f' % (avg_eval_length, avg_eval_lr))
        print('Avg lead3 quality reward: ', avg_lead3_qr)
        print('Avg lead3 length: %.4f, reward: %.4f' % (avg_lead3_length, avg_lead3_lr))
        return avg_eval_qr, avg_eval_lr, avg_lead3_qr, avg_lead3_lr

    
    
    
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=3)
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
    parser.add_argument('--std_rouge', action='store_true')
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