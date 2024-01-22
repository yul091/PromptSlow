import os
import sys 
sys.dont_write_bytecode = True
sys.setrecursionlimit(2000)  # Increase the limit, 2000 is just an example
import time
from tqdm import tqdm
from argparse import Namespace
from typing import Optional, List, Tuple, Dict, Union, Any
import torch 
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    pipeline,
    AutoConfig, 
    AutoTokenizer, 
    T5Tokenizer,
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)
# import pdb # debug breakpoint
from rouge import Rouge
from Dataset import prompt_dataset, get_dataloader
from test_llama import extract_ans, parse_pred_ans
from huggingface_api import generation_pipeline, add_model_args


class PromptTransformer:
    def __init__(
        self, 
        args: Namespace,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = 'cuda:0',
    ):
        self.args = args
        self.task = args.task
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.my_model_name = args.my_model_name
        self.device = device
        self.max_length = args.max_length
        self.max_new_tokens = args.max_new_tokens
        self.saving_step = args.saving_step
        self.output_dir = args.output_dir
        self.n_train_samples = args.n_train_samples
        self.n_test_samples = args.n_test_samples
        self.use_greedy_baseline = args.use_greedy_baseline
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
            
        self.config = AutoConfig.from_pretrained(self.my_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.my_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.my_model_name)

        os.makedirs(self.output_dir, exist_ok=True)
         
        self.optimizer = optimizer   
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)    
            
        self.instruction, self.demonstrations, self.prefix, self.train_dataset, self.test_dataset = prompt_dataset(task=self.task)
        if self.n_train_samples > 0:
            n_train_samples = min(self.n_train_samples, len(self.train_dataset))
            self.train_dataset = self.train_dataset.select(range(n_train_samples))
            
        if self.n_test_samples > 0:
            n_test_samples = min(self.n_test_samples, len(self.test_dataset))
            self.test_dataset = self.test_dataset.select(range(n_test_samples))
            
        # Data collator
        label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )
        
        self.train_dataloader = get_dataloader(
            self.train_dataset, 
            self.tokenizer, 
            batch_size=self.batch_size,
            padding=True,
            ignore_pad_token_for_loss=self.ignore_pad_token_for_loss,
            collate_fn=data_collator,
        )
        
        self.test_dataloader = get_dataloader(
            self.test_dataset, 
            self.tokenizer, 
            batch_size=self.batch_size,
            padding=True,
            ignore_pad_token_for_loss=self.ignore_pad_token_for_loss,
            collate_fn=data_collator,
        )
        
        
    def calculate_rewards(
        self,
        answer: str,
        response: str,
    ) -> Tuple[float, float]:
        
        # Quality reward: average of F-meansure of ROUGE-1, ROUGE-2, and ROUGE-L
        rouge = Rouge()
        if not response or not answer:
            scores = [{'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}]
        else:
            scores = rouge.get_scores([response], [answer])
        rouge_score = (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3
        
        # Length reward
        length_reward = len(self.tokenizer.tokenize(answer)) / len(self.tokenizer.tokenize(response))
    
        return length_reward, rouge_score
        
        
    def prepare_inputs(self, batch: Dict[str, Union[Any, torch.Tensor]]):
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        
    def training_step(
        self,
        batch: Dict[str, Union[Any, torch.Tensor]],
    ):
        self.model.train()
        args = self.args
        answers = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True) # (B)
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')  # Use .get to handle cases where it might not exist       
        
        # Generate new prompts
        with torch.no_grad():
            sampled_hyps = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                max_length=self.max_length, 
                do_sample=True,
            ) # (B X T)

        sampled_queries = self.tokenizer.batch_decode(sampled_hyps, skip_special_tokens=True) # (B)
        sampled_prompts = [self.instruction + self.demonstrations + self.prefix + q for q in sampled_queries]
        sampled_responses = []
        for sampled_prompt in sampled_prompts:
            # print("new_prompt: ", sampled_prompt)
            args.message = sampled_prompt
            sampled_response = generation_pipeline(args)
            sampled_responses.append(sampled_response)
            
        # Calculate rewards
        rewards = [self.calculate_rewards(ans, resp) for ans, resp in zip(answers, sampled_responses)]
        length_rewards, quality_rewards = zip(*rewards)
        avg_length_reward = sum(length_rewards) / len(length_rewards)
        avg_quality_reward = sum(quality_rewards) / len(quality_rewards)
        
        if self.use_greedy_baseline:
            with torch.no_grad():
                greedy_hyps = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    max_length=self.max_length, 
                    do_sample=False,
                )
            
            greedy_queries = self.tokenizer.batch_decode(greedy_hyps, skip_special_tokens=True) # (B)
            greedy_prompts = [self.instruction + self.demonstrations + self.prefix + q for q in greedy_queries] 
            greedy_responses = []
            for greedy_prompt in greedy_prompts:
                args.message = greedy_prompt
                greedy_response = generation_pipeline(args)
                greedy_responses.append(greedy_response)
            
            # Greedy rewards
            greedy_rewards = [self.calculate_rewards(ans, resp) for ans, resp in zip(answers, greedy_responses)]
        
        # Implement policy update logic based on rewards
        # This involves backpropagation and optimizer steps
        # Generate outputs and calculate log-probabilities of the actions taken
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=sampled_hyps, # (B X T)
        )
        log_prob = outputs.loss  # Negative log-likelihood

        # Weight log_probs by rewards and calculate loss
        weighted_loss = 0
        if self.use_greedy_baseline:
            for (l_r, q_r), (base_l_r, base_q_r) in zip(rewards, greedy_rewards):
                weighted_loss += - log_prob * (l_r + q_r - base_l_r - base_q_r) # force larger rewards to be more important
        else:
            for l_r, q_r in rewards:
                weighted_loss += - log_prob * (l_r + q_r)
        weighted_loss /= len(rewards)

        # Perform backpropagation
        weighted_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # pdb.set_trace()

        return weighted_loss.item(), avg_length_reward, avg_quality_reward
    

    def train(self):
        self.model.to(self.device)
        global_step = 0
        model_n = self.model.__class__.__name__
        
        for epoch in range(self.num_epochs):
            
            self.save_file = f"{self.output_dir}/{args.model_path.split('/')[-1]}_{self.task}_{model_n}_epoch-{epoch+1}.txt"
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for step, batch in pbar:
                batch = self.prepare_inputs(batch)
                weighted_loss, avg_length_reward, avg_quality_reward = self.training_step(batch)
                global_step += 1
                pbar.set_description(f'[step {global_step}] loss: {weighted_loss:.4f}, length reward: {avg_length_reward:.4f}, quality reward: {avg_quality_reward:.4f}')
                if global_step % self.saving_step == 0:
                    # Save model
                    self.model.save_pretrained(f'{self.output_dir}/checkpoint-{global_step}')
                
            metrics = self.evaluate(self.test_dataloader)
            print(f'[epoch {epoch}]: \n{metrics}')
            # Save metrics
            with open(f'{self.output_dir}/metrics_{self.task}.txt', 'a') as f:
                f.write(f'[epoch {epoch}]: \n{metrics}\n')
            
            
    
    @torch.no_grad()
    def eval_step(
        self, 
        batch: Dict[str, Union[Any, torch.Tensor]], 
        metrics: Dict[str, Any], 
    ):
        self.model.eval()
        args = self.args
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')  # Use .get to handle cases where it might not exist
        original_queries = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        print("original_queries: ", original_queries)
        answers = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        # Generate new responses
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_length=self.max_length, 
            do_sample=False,
        ) # (B X T)
        new_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)   
        print("new_queries: ", new_queries)
        print("answers: ", answers)
        
        responses = []
        for new_query in new_queries:
            # print("new_prompt: ", new_prompt)
            new_prompt = self.instruction + self.demonstrations + self.prefix + new_query
            args.message = new_prompt
            response = generation_pipeline(args)
            responses.append(response)
        print("new_responses: ", responses)

        # Calculate rouge scores and length
        rouge = Rouge()
        
        for original_query, query, answer, response in zip(original_queries, new_queries, answers, responses):
            output_tokens = self.tokenizer.tokenize(response)
            
            if self.task == 'ICL':
                ans_, residual = extract_ans(response)
                with open(self.save_file, 'a') as fd:
                    fd.write("Q:\n%s\nQ':\n%s\nA_model:\n%s\nA:\n%s\n\n" % (
                        original_query,
                        query, 
                        ans_.replace("Q:", "").replace("A:", ""), 
                        answer,
                    ))
            else:
                with open(self.save_file, 'a') as fd:
                    fd.write("Q:\n%s\nQ'\n%s\nA_model:\n%s\nA:\n%s\n\n" % (
                        original_query,
                        query, 
                        response, 
                        answer,
                    ))       
                scores: dict = rouge.get_scores([response], [answer])
                for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                    metrics[metric].append(scores[0][metric]['f'])
                
            metrics['length'].append(len(output_tokens))
        
        # pdb.set_trace()
            
            
    def evaluate(
        self, 
        dataloader: DataLoader,
        use_transformation: bool = True,
    ):
        self.model.to(self.device)
        
        metrics = {
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'length': [],
        }
        
        start = time.time()
        if use_transformation:
            for batch in tqdm(dataloader):
                batch = self.prepare_inputs(batch)
                self.eval_step(batch, metrics)
            if self.task == 'ICL':  
                questions, ans_pred, ans_gold, num_q, acc = parse_pred_ans(self.save_file)
                metrics['EM'] = [float(acc / num_q)]
        else:
            save_file = f"{self.output_dir}/{args.model_path.split('/')[-1]}_{self.task}.txt"
            for instance in tqdm(self.test_dataset, total=len(self.test_dataset)):
                q = instance['query']
                a = instance['reference']
                prompt = self.instruction + self.demonstrations + self.prefix + q
                self.args.message = prompt
                response = generation_pipeline(self.args)
                
                if self.task == 'ICL':
                    ans_, residual = extract_ans(response)
                    with open(save_file, 'a') as fd:
                        fd.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (
                            q, 
                            ans_.replace("Q:", "").replace("A:", ""), 
                            a,
                        ))
                else:   
                    with open(save_file, 'a') as fd:
                        fd.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (
                            q, 
                            response, 
                            a,
                        ))
                
                    # print("response: ", response)
                    rouge = Rouge()
                    if not response or not a:
                        scores = [{'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}]
                    else:
                        scores = rouge.get_scores([response], [a])
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        metrics[metric].append(scores[0][metric]['f'])
                        
                output_tokens = self.tokenizer.tokenize(response)
                metrics['length'].append(len(output_tokens))
                
            if self.task == 'ICL':  
                questions, ans_pred, ans_gold, num_q, acc = parse_pred_ans(save_file)
                metrics['EM'] = [float(acc / num_q)]
        end = time.time()
            
        # Average
        for metric in metrics:
            metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
            
        metrics['overhead'] = end - start
        metrics['overhead_per_sample'] = metrics['overhead'] / len(self.test_dataset)
            
        return metrics
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--my_model_name", type=str, default="t5-small")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    # Training arguments 
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--saving_step", type=int, default=100)
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    parser.add_argument("--use_greedy_baseline", action="store_true")
    # Data arguments
    parser.add_argument("--task", type=str, default="conversational")
    parser.add_argument("--n_train_samples", type=int, default=-1)
    parser.add_argument("--n_test_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    # generation_pipeline(args)
    pt = PromptTransformer(args, device='cuda')
    metrics = pt.evaluate(pt.test_dataloader, use_transformation=False)
    print("Metrics without transformation: {}".format(metrics))
    # Save metrics
    with open(f'{pt.output_dir}/metrics_{args.task}.txt', 'w') as f:
        f.write(f'[before transformation]: \n{metrics}\n')
    pt.train()
        
    