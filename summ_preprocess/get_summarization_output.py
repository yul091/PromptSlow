from datasets import load_dataset
import argparse
import torch
from fastchat.model import add_model_args, load_model, get_conversation_template
import json

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

def create_cnndm_dataset():
    cnn = load_dataset("cnn_dailymail", '3.0.0')
    print(cnn)
    # Standardize the keys (text -> query, summary -> reference)
    cnn = cnn.rename_column("article", "query")
    cnn = cnn.rename_column("highlights", "reference")
    cnn_train = cnn["train"].select(range(1000))
    cnn_test = cnn["test"].select(range(1000))
    return cnn_train, cnn_test

def create_xsum_dataset():
    xsum = load_dataset("xsum")
    print(xsum)
    # Standardize the keys (text -> query, summary -> reference)
    xsum = xsum.rename_column("document", "query")
    xsum = xsum.rename_column("summary", "reference")
    xsum_train = xsum["train"].select(range(1000))
    xsum_test = xsum["test"].select(range(1000))
    return xsum_train, xsum_test

def create_pubmed_dataset():
    pubmed = load_dataset("ccdv/pubmed-summarization")
    print(pubmed)
    # Standardize the keys (text -> query, summary -> reference)
    pubmed = pubmed.rename_column("article", "query")
    pubmed = pubmed.rename_column("abstract", "reference")
    pubmed_train = pubmed["train"].select(range(1000))
    pubmed_test = pubmed["test"].select(range(1000))
    return pubmed_train, pubmed_test

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
    # Get rid of empty queries or references
    sharegpt = sharegpt.filter(lambda x: len(x['query']) > 0 or len(x['reference']) > 0)
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

@torch.inference_mode()
def generation_pipeline(args, model, tokenizer):
    model = model.bfloat16()

    # Build the prompt with a conversation template
    msg = args.message
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs

if __name__ == "__main__":
    # train_dataset, test_dataset = create_sharegpt_dataset()
    # print(train_dataset)
    # print(test_dataset)
    # print(train_dataset[0])
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--dataset', type=str, default='billsum')
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2
    
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    if args.dataset == 'xsum': 
        train_dataset, test_dataset = create_xsum_dataset()
    elif args.dataset == 'cnn':
        train_dataset, test_dataset = create_cnndm_dataset()
    elif args.dataset == 'pubmed':
        train_dataset, test_dataset = create_pubmed_dataset()
    elif args.dataset == 'sharegpt':
        train_dataset, test_dataset = create_sharegpt_dataset()

    total_result= []
    from tqdm import tqdm
    for item in tqdm(train_dataset, total=len(train_dataset)):
        cur_example = {}
        document = item['query']
        reference = item['reference']
        args.message = args.prompt.replace('\\n', '\n') + document
        result = generation_pipeline(args, model, tokenizer)
        cur_example['input'] = args.message
        cur_example['output'] = result
        cur_example['reference'] = reference
        total_result.append(cur_example)
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(total_result, f, indent=4, ensure_ascii=False)
