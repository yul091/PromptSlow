from __future__ import print_function
import sys
sys.dont_write_bytecode = True
import random
from helper import Document
import numpy as np
import torch
import argparse
from typing import List, Dict, Any, Union, Tuple
from torch.autograd import Variable
from transformers import PreTrainedTokenizerFast
from rougefonc import from_summary_index_compute_rouge


method = 'herke'


def return_summary_index(
    probs_numpy: np.ndarray, 
    probs_torch: np.ndarray, 
    sample_method: str = "greedy", 
    max_num_of_sents: int = 3,
) -> (np.ndarray, torch.FloatTensor):
    """
    probs: numpy array of the probablities for all sentences in the doc
    sample_method: greey or sample
    max_num_of_sents: max num of sents to be selected
    -> a list of index for the selected sents, and the corresponding total loss
    """
    assert isinstance(sample_method, str)
    
    if max_num_of_sents <= 0:
        if sample_method == "sample":
            l = np.random.binomial(1, probs_numpy)
        elif sample_method == "greedy":
            l = [1 if prob >= 0.5 else 0 for prob in probs_numpy]
        summary_index = np.nonzero(l)[0]
        
    else:
        if sample_method == "sample":
            # print("probs torch: ", probs_torch)
            probs_torch = probs_torch.squeeze(-1)
            # assert len(probs_torch.size()) == 1 
            if len(probs_torch.size()) == 1:
                return np.arange(len(probs_torch)), probs_torch.sum()
            
            if method == 'original':
                # original method
                probs_clip = probs_numpy * 0.8 + 0.1
                # print("sampling the index for the summary")
                index = range(len(probs_clip)) # (B)
                probs_clip_norm = probs_clip / sum(probs_clip)
                summary_index = np.random.choice(
                    index, 
                    max_num_of_sents, 
                    replace=False,
                    p=np.reshape(probs_clip_norm, len(probs_clip_norm)),
                )
                p_summary_index = probs_numpy[summary_index]
                sorted_idx = np.argsort(p_summary_index)[::-1]
                summary_index = summary_index[sorted_idx]
                loss = 0.
                for idx in index:
                    if idx in summary_index:
                        loss += probs_torch[idx].log()
                    else:
                        loss += (1 - probs_torch[idx]).log()
                        
            elif method == 'herke':
                # herke's method
                summary_index = []
                epsilon = 0.1
                mask = Variable(torch.ones(probs_torch.size()).cuda(), requires_grad=False)
                loss_list = []
                for i in range(max_num_of_sents):
                    p_masked = probs_torch * mask
                    if random.uniform(0, 1) <= epsilon:  # explore
                        selected_idx = torch.multinomial(mask, 1)
                    else:
                        selected_idx = torch.multinomial(p_masked, 1)
                    loss_i = (epsilon / mask.sum() + (1 - epsilon) * p_masked[selected_idx] / p_masked.sum()).log()
                    loss_list.append(loss_i)
                    mask = mask.clone()
                    mask[selected_idx] = 0
                    summary_index.append(selected_idx)

                summary_index = torch.cat(summary_index, dim=0)
                summary_index = summary_index.data.cpu().numpy()
                loss = sum(loss_list)
                
        elif sample_method == "greedy":
            loss = 0
            summary_index = np.argsort(np.reshape(probs_numpy, len(probs_numpy)))[-max_num_of_sents:]
            summary_index = summary_index[::-1]

    # summary_index.sort()
    return summary_index, loss


class ReinforceReward:
    def __init__(
        self, 
        args: argparse.Namespace,
        tokenizer: PreTrainedTokenizerFast,
        rouge_metric: str = "all", 
        b: int = 5, 
        rl_baseline_method: str = "greedy", 
        loss_method: int = 1,
    ):
        """
        :param rouge_metric:
        :param b:
        :param rl_baseline: "greedy", "global_avg", "batch_avg", "batch_med", None
        """
        self.args = args
        self.probs_torch = None
        self.probs_numpy = None
        self.max_num_of_sents = None
        self.doc = None

        self.global_avg_quality_reward = 0.
        self.global_avg_length_reward = 0.
        self.train_examples_seen = 0.

        self.std_rouge = False
        self.rouge_metric = rouge_metric
        self.rl_baseline_method = rl_baseline_method
        self.b = b  # batch_size
        self.loss_method = loss_method
        self.tokenizer = tokenizer
        
        
    def update_data_instance(self, probs: np.ndarray, doc: Document, max_num_of_sents: str = 3):
        # self.probs_torch = probs
        # self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero
        self.probs_torch = probs * 0.9999 + 0.00005  # just make sure no zero
        probs_numpy = probs.data.cpu().numpy() # (B, 1)
        self.probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
        self.doc = doc
        self.max_num_of_sents = min(len(probs_numpy), max_num_of_sents)
        
        
    def generate_index_list_and_loss(self, sample_method: str = "sample") -> (np.ndarray, torch.FloatTensor):
        """
        :param sample_method: "lead3,leadk,sample, greedy"
        :return: return a list of sentence indexes for next step of computation
        """
        if sample_method == "lead3":
            return range(3), 0
        elif sample_method == "lead_oracle":
            return range(self.max_num_of_sents), 0
        else:  # either "sample" or "greedy" based on the prob_list
            return return_summary_index(
                self.probs_numpy, 
                self.probs_torch,
                sample_method=sample_method, 
                max_num_of_sents=self.max_num_of_sents,
            )
            
    def sample_batch(self, b: int) -> List[Tuple[np.ndarray, torch.FloatTensor]]:
        batch_index_and_loss_lists = [self.generate_index_list_and_loss() for _ in range(b)]
        return batch_index_and_loss_lists
    
    
    def generate_reward(self, summary_index_list: np.ndarray, max_num_of_bytes: int = -1) -> Tuple[float, float]:
        return from_summary_index_compute_rouge(
            self.args, 
            self.doc, 
            summary_index_list,
            self.tokenizer,
            std_rouge=self.std_rouge,
            rouge_metric=self.rouge_metric,
            max_num_of_bytes=max_num_of_bytes,
        )
    
    
    def compute_baseline(self, batch_rewards: List[Tuple[float, float]]) -> Tuple[float, float]:
        def running_avg(t, old_mean, new_score):
            return (t - 1) / t * old_mean + new_score / t

        quality_rewards, length_rewards = zip(*batch_rewards)
        batch_avg_quality_reward = np.mean(quality_rewards)
        batch_avg_length_reward = np.mean(length_rewards)
        batch_median_quality_reward = np.median(quality_rewards)
        batch_median_length_reward = np.median(length_rewards)
        self.global_avg_quality_reward = running_avg(
            self.train_examples_seen, 
            self.global_avg_quality_reward,
            batch_avg_quality_reward,
        )
        self.global_avg_length_reward = running_avg(
            self.train_examples_seen, 
            self.global_avg_length_reward, 
            batch_avg_length_reward,
        )
        if self.rl_baseline_method == "batch_avg":
            return batch_avg_quality_reward, batch_avg_length_reward
        if self.rl_baseline_method == "batch_med":
            return batch_median_quality_reward, batch_median_length_reward
        elif self.rl_baseline_method == "global_avg":
            return self.global_avg_quality_reward, self.global_avg_length_reward
        elif self.rl_baseline_method == "greedy":
            summary_index_list, _ = self.generate_index_list_and_loss("greedy")
            return self.generate_reward(summary_index_list)
        else:  # none
            return 0, 0
        
        
    def generate_batch_loss(
        self, 
        batch_index_and_loss_lists: List[Tuple[np.ndarray, torch.FloatTensor]], 
        batch_rewards: List[Tuple[float, float]], 
        base_quality_reward: float,
        base_length_reward: float,
    ) -> torch.FloatTensor:
        loss_list = [
            batch_index_and_loss_lists[i][1] * ((base_quality_reward - batch_rewards[i][0] + base_length_reward - batch_rewards[i][1]) / (base_quality_reward + base_length_reward + 1e-9))
            for i in range(len(batch_rewards))
        ]
        avg_loss = sum(loss_list) / (float(len(loss_list)) + 1e-8)
        return avg_loss
        

    def train(
        self, 
        probs: np.ndarray, 
        doc: Document, 
        max_num_of_sents: int = 3, 
        max_num_of_bytes: int = -1, 
        prt: bool = False,
    ) -> Tuple[torch.FloatTensor, float, float]:
        """
        return: training_loss_of_the current example
        """
        self.update_data_instance(probs, doc, max_num_of_sents)
        self.train_examples_seen += 1
        batch_index_and_loss_lists = self.sample_batch(self.b)
        batch_rewards = [
            self.generate_reward(summary_index, max_num_of_bytes)
            for (summary_index, loss) in batch_index_and_loss_lists
        ]
        # print('Sample rewards: ', batch_rewards)

        base_quality_r, base_length_r = self.compute_baseline(batch_rewards)
        loss = self.generate_batch_loss(
            batch_index_and_loss_lists, 
            batch_rewards, 
            base_quality_reward=base_quality_r,
            base_length_reward=base_length_r,
        )
        # print(loss)

        greedy_index_list, _ = self.generate_index_list_and_loss("greedy")
        greedy_quality_reward, greedy_length_reward = self.generate_reward(greedy_index_list, max_num_of_bytes)
        # print("Greedy rewards:", greedy_reward)

        if prt:
            print('Batch rewards:', np.array(batch_rewards))
            print('Greedy quality rewards:', np.array(greedy_quality_reward))
            print('Greedy length rewards:', np.array(greedy_length_reward))
            print('Baseline qualtiy rewards:', np.array(base_quality_r))
            print('Baseline length rewards:', np.array(base_length_r))

            lead_index_list, _ = self.generate_index_list_and_loss("lead3")
            lead_reward = self.generate_reward(lead_index_list, max_num_of_bytes)
            print('Lead3 rewards:', np.array(lead_reward))

        return loss, greedy_quality_reward, greedy_length_reward
    

    def validate(self, probs: np.ndarray, doc: Document, max_num_of_sents: int = 3) -> Tuple[float, float]:
        """
        :return: training_loss_of_the current example
        """
        self.update_data_instance(probs, doc, max_num_of_sents)
        summary_index_list, _ = self.generate_index_list_and_loss("greedy")
        reward_tuple = from_summary_index_compute_rouge(
            self.args,
            self.doc, 
            summary_index_list,
            self.tokenizer,
            std_rouge=self.std_rouge, 
            rouge_metric="all",
        )
        return reward_tuple

    
    def generate_summary(self, summary_index_list: np.ndarray) -> List[str]:
        return [self.doc.query[i] for i in summary_index_list]


    

    


if __name__ == '__main__':
    pass