import sys 
sys.dont_write_bytecode = True
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Union, Any
import torch 
from torch.optim import AdamW
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    T5Tokenizer,
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)
from rouge import Rouge
from dataset import prompt_dataset, get_dataloader


class PromptTransformer:
    def __init__(
        self, 
        model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_token: Optional[str] = None,
        task: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: Optional[float] = 5e-5,
        batch_size: Optional[int] = 1,
        max_length: Optional[int] = 2096,
        max_new_tokens: Optional[int] = 2096,
        ignore_pad_token_for_loss: Optional[bool] = True,
        num_epochs: Optional[int] = 1,
        device: Optional[str] = 'cuda:0',
    ):
        if model_name is None:
            model_name = 't5-base'
            
        # self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.num_epochs = num_epochs
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        if llm_model_name is not None:
            llm_model_name = 'meta-llama/Llama-2-7b-hf'
            
        if llm_token is not None:
            llm_token = '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'
        self.llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=llm_token)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token=llm_token)
        
        if task is None:
            task = "summarization"
         
        self.optimizer = optimizer   
        if optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=lr)    
        
            
        self.instruction, self.demonstrations, self.prefix, self.train_dataset, self.test_dataset = prompt_dataset(task=task)
        
        # Data collator
        label_pad_token_id = -100 if ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )
        
        self.train_dataloader = get_dataloader(
            self.train_dataset, 
            self.tokenizer, 
            batch_size=batch_size,
            instruction=self.instruction,
            demonstrations=self.demonstrations,
            prefix=self.prefix,
            max_length=max_length,
            padding=True,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            collate_fn=data_collator,
        )
        
        self.test_dataloader = get_dataloader(
            self.test_dataset, 
            self.tokenizer, 
            batch_size=batch_size,
            instruction=self.instruction,
            demonstrations=self.demonstrations,
            prefix=self.prefix,
            max_length=max_length,
            padding=True,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            collate_fn=data_collator,
        )
        
        
    def calculate_rewards(
        self,
        answer: str,
        response: str,
    ) -> Tuple[float, float]:
        
        # Quality reward: average of F-meansure of ROUGE-1, ROUGE-2, and ROUGE-L
        rouge = Rouge()
        scores: dict = rouge.get_scores([response], [answer])
        rouge_score = (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3
        # rouge_score = RougeTest_rouge(answer, response, rouge_metric='avg_f')
        
        # Length reward
        # length_reward = len(answer) / len(response)
        length_reward = len(self.tokenizer.tokenize(answer)) / len(self.tokenizer.tokenize(response))
    
        return length_reward, rouge_score
        

    def policy_update(
        self, 
        tokenizer: T5Tokenizer,
        model: T5ForConditionalGeneration, 
        inputs: Dict[str, Union[Any, torch.Tensor]], 
        new_prompts: List[str], 
        rewards: List[Tuple[float, float]], 
        optimizer: torch.optim.Optimizer,
    ):
        # Convert transformed prompts to tensors
        transformed_inputs = tokenizer(
            new_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
        )
        transformed_inputs = self.prepare_inputs(transformed_inputs)

        # Generate outputs and calculate log-probabilities of the actions taken
        outputs = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=transformed_inputs['input_ids'],
        )
        log_prob = outputs.loss  # Negative log-likelihood

        # Weight log_probs by rewards and calculate loss
        weighted_loss = 0
        for i, (l_r, q_r) in enumerate(rewards):
            weighted_loss += - log_prob * (l_r + q_r) # force larger rewards to be more important
        weighted_loss /= len(rewards)

        # Perform backpropagation
        weighted_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return weighted_loss.item()
        
        
    def prepare_inputs(self, batch: Dict[str, Union[Any, torch.Tensor]]):
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        
    def training_step(
        self,
        batch: Dict[str, Union[Any, torch.Tensor]],
        
    ):
        self.model.train()
        self.llm_model.eval()
        
        # Generate new responses
        print(f"{self.model.__class__.__name__} transforming prompt ...")
        with torch.no_grad():
            outputs = self.model.generate(**batch, max_length=self.max_length) # (batch_size, max_length)
        new_prompts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) # (batch_size)
        answers = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True) # (batch_size)
        
        print(f"{self.llm_model.__class__.__name__} generating response ...")
        llm_inputs = self.llm_tokenizer(
            new_prompts, 
            return_tensors='pt',
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            llm_outputs = self.llm_model.generate(**llm_inputs, max_new_tokens=self.max_new_tokens)
        new_responses = self.llm_tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)

        print("Calculate rewards ...")
        # Calculate rewards
        rewards = [self.calculate_rewards(ans, resp) for ans, resp in zip(answers, new_responses)]
        length_rewards, quality_rewards = zip(*rewards)
        avg_length_reward, avg_quality_reward = sum(length_rewards) / len(length_rewards), sum(quality_rewards) / len(quality_rewards)

        # TODO: Implement policy update logic based on rewards
        # This involves backpropagation and optimizer steps
        print("Policy update ...")
        weighted_loss = self.policy_update(
            self.tokenizer,
            self.model,
            batch,
            new_prompts,
            rewards,
            self.optimizer,
        )
        return weighted_loss, avg_length_reward, avg_quality_reward
    

    def train(self):
        self.model.to(self.device)
        self.llm_model.to(self.device)
        
        for epoch in range(self.num_epochs):
            pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            for batch in pbar:
                batch = self.prepare_inputs(batch)
                weighted_loss, avg_length_reward, avg_quality_reward = self.training_step(batch)
                pbar.set_description(f'[epoch {epoch}] loss: {weighted_loss:.4f}, length reward: {avg_length_reward:.4f}, quality reward: {avg_quality_reward:.4f}')
                
            metrics = self.evaluate(self.model, self.test_dataloader)
            print(f'[epoch {epoch}]: \n{metrics}')
                
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Union[Any, torch.Tensor]], metrics: Dict[str, Any]):
        self.model.eval()
        # Generate new responses
        outputs = self.model.generate(**batch, max_length=self.max_length) # (batch_size, max_length)
        new_prompts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        llm_inputs = self.llm_tokenizer(new_prompts, return_tensors='pt').to(batch.device)
        llm_outputs = self.llm_model.generate(**llm_inputs, max_new_tokens=self.max_new_tokens)
        new_responses = self.llm_tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)

        # Calculate rouge scores and length
        rouge = Rouge()
        for answer, output, response in zip(answers, llm_outputs, new_responses):
            # rouge_scores = RougeTest_rouge(answer, response, rouge_metric='all')
            scores: dict = rouge.get_scores([response], [answer])
            for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                metrics[metric].append(scores[0][metric]['f'])
                
            metrics['length'].append(output.shape[0])
            
            
    def evaluate(self):
        self.model.to(self.device)
        
        metrics = {
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'length': [],
        }
        
        for batch in self.test_dataloader:
            batch = self.prepare_inputs(batch)
            self.eval_step(batch, metrics)
            
        return metrics
    
    
    
if __name__ == '__main__':
    pt = PromptTransformer()
    pt.train()
        
    