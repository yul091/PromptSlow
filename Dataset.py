import sys 
sys.dont_write_bytecode = True
# import numpy as np
from typing import List, Optional, Callable
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


def prompt_dataset(task: str):
    if task == 'summarization':
        instruction = ""
        demonstrations = ""
        prefix = "\n\nSummarize the following article:\n"
        train_dataset, test_dataset = create_billsum_dataset()
        
    elif task == "ICL":
        instruction = "Please reference the following examples to answer the math question:\n"
        demonstrations = open("./prompt_hardest.txt").read()
        prefix = "\n\nQuestion: "
        train_dataset, test_dataset = create_gsm8k_dataset()
        
    elif task == "conversational":
        instruction = ""
        demonstrations = ""
        prefix = ""
        train_dataset, test_dataset = create_sharegpt_dataset()
    else:
        raise ValueError(task)
    
    return instruction, demonstrations, prefix, train_dataset, test_dataset
    

def create_billsum_dataset():
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2) 
    print(billsum)
    # Standardize the keys (text -> query, summary -> reference)
    billsum = billsum.rename_column("text", "query")
    billsum = billsum.rename_column("summary", "reference")
    billsum_train = billsum["train"] # (989) text, summary, title
    billsum_test = billsum["test"] # (248) text, summary, title
    return billsum_train, billsum_test


def create_gsm8k_dataset():
    gsm8k = load_dataset("gsm8k", "main")
    print(gsm8k)
    # Standardize the keys (question -> query, answer -> reference)
    gsm8k = gsm8k.rename_column("question", "query")
    gsm8k = gsm8k.rename_column("answer", "reference")
    gsm8k_train = gsm8k["train"] # (7473) question, answer
    gsm8k_test = gsm8k["test"] # (1319) question, answer
    return gsm8k_train, gsm8k_test


def create_sharegpt_dataset():
    
    def process_conversation(examples):
        queries = []
        references = []
        # Ensure there are at least two turns in the conversation
        for conversation in  examples['conversations']:
            if len(conversation['value']) >= 2:
                queries.append(conversation['value'][0])  # First turn for query
                references.append(conversation['value'][1])  # Second turn for reference
            else:
                # Handle conversations with less than 2 turns (if any)
                queries.append("")
                references.append("")

        examples['query'] = queries
        examples['reference'] = references
        return examples
    
    sharegpt = load_dataset("liyucheng/ShareGPT90K") # 90665
    # random sample 1000 examples
    # idx = np.random.choice(len(sharegpt['train']), 5000, replace=False)
    sharegpt['train'] = sharegpt['train'].select(range(1000))
    # sharegpt = load_dataset("liyucheng/sharegpt-500") # 500
    # Standardize the keys (text -> query, summary -> reference)
    column_names = sharegpt['train'].column_names
    sharegpt = sharegpt.map(
        process_conversation,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on llm dataset",
    )
    try:
        sharegpt = sharegpt.train_test_split(test_size=0.2)
        sharegpt_train = sharegpt["train"] # (72532) text, summary
        sharegpt_test = sharegpt["test"] # (18133) text, summary
    except:
        train_size = 0.8
        sharegpt_train = sharegpt['train'].select(range(int(train_size * len(sharegpt['train']))))
        sharegpt_test = sharegpt['train'].select(range(int(train_size * len(sharegpt['train'])), len(sharegpt['train'])))
    print(sharegpt)
    return sharegpt_train, sharegpt_test


def get_dataloader(
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizer, 
    batch_size: Optional[int] = 32,
    instruction: Optional[str] = '',
    demonstrations: Optional[str] = '',
    prefix: Optional[str] = '',
    max_length: Optional[int] = None,
    padding: Optional[str] = None,
    ignore_pad_token_for_loss: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    column_names = dataset.column_names
    
    def preprocess_function(examples):
        # Tokenize the texts
        inputs = [instruction + demonstrations + prefix + query for query in examples["query"]]
        targets = [reference for reference in examples["reference"]]
        model_inputs = tokenizer(inputs, padding=padding, truncation=True)
        labels = tokenizer(targets, padding=padding, truncation=True)
        # print("Input sequence lengths: ", [len(input_ids) for input_ids in model_inputs["input_ids"]])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on llm dataset",
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
    )
    
    return dataloader




if __name__ == "__main__":
    task = "conversational" # "summarization", "ICL", "conversational"
    instruction, demonstrations, prefix, train_dataset, test_dataset = prompt_dataset(task=task)
    print(train_dataset)
    print('instance[0]: ', train_dataset[0])