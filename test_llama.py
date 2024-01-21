import os
import re
import sys
sys.dont_write_bytecode = True
import torch
from Dataset import prompt_dataset
from llmlingua import PromptCompressor
from huggingface_api import generation_pipeline, add_model_args

def test_llama(args):
    # Instantiate tokenizer and model
    # args.model_path = 'meta-llama/Llama-2-7b-hf'
    # args.model_path = "/data/yli927/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
    # args.token = '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'
    args.device = 'cuda'
    args.gpus="0,1"
    
    # In-Context Learning (ICL)
    # !wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    instruction, demonstrations, prefix, train_dataset, test_dataset = prompt_dataset("ICL")
    question, answer = test_dataset['query'][0], test_dataset['reference'][0]
    prompt = instruction + demonstrations + prefix + question
    print("\nPrompt:", prompt)
    args.message = prompt
    outputs = generation_pipeline(args)
    print("\nResponse:", outputs)
    print("\nAnswer:", answer)
    
    # Summarization
    instruction, demonstrations, prefix, train_dataset, test_dataset = prompt_dataset("summarization")
    question, answer = test_dataset['query'][0], test_dataset['reference'][0]
    prompt = instruction + demonstrations + prefix + question
    print("\nPrompt:", prompt)
    args.message = prompt
    outputs = generation_pipeline(args)
    print("\nResponse:", outputs)
    print("\nAnswer:", answer)


# Evaluate
def extract_ans(ans_model: str):
    ans_model = ans_model.split("\n")
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if "answer is" in al:
            break
    residual = list(ans_model[li + 1 :])
    ans = "\n".join(ans)
    residual = "\n".join(residual)
    return ans, residual

def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = "none"
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        l = l.replace(",", "")
        if l.startswith("Q: "):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = "q"
            q = l
            num_q += 1
        elif l.startswith("A_model:"):
            current_mode = "am"
            am = l
        elif l.startswith("A:"):
            current_mode = "a"
            a = l
        else:
            if current_mode == "q":
                q += l
            elif current_mode == "am":
                am += l
            elif current_mode == "a":
                a += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if test_answer(am, a):
        acc += 1
    print("num_q %d correct %d ratio %.4f" % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


def get_result(text: str):
    pattern = "\d*\.?\d+"
    res = re.findall(pattern, text)
    return res[-1] if res else ""


def test_answer(pred_str, ans_str):
    pred, gold = get_result(pred_str), get_result(ans_str)
    return pred == gold

# os.makedirs("outputs", exist_ok=True)
# i = 0
# for q, a in tqdm(zip(gsm8k_test['question'], gsm8k_test['answer']), 
#                            total=len(gsm8k_test['question'])):
#     instruction = "Please reference the following examples to answer the math question,\n"
#     prompt = instruction + prompt_complex + "\n\nQuestion: " + q + "\n"
#     # ans_model = response["choices"][0]["text"]
#     inputs = tokenizer(q, return_tensors='pt').to(device)
#     response = model.generate(**inputs)
#     ans_model = tokenizer.decode(output[0], skip_special_tokens=True)
#     ans_, residual = extract_ans(ans_model)
#     # with open('outputs/test_gpt_3.5_turbo_LLMLingua_174.txt', 'a') as fd:
#     with open('outputs/test_blenderbot_3B.txt', 'a') as fd:
#         fd.write("Q: %s\nA_model:\n%s\nA:\n%s\n\n" % (q, ans_.replace("Q:", "").replace("A:", ""), a))
#     i += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2
        
    test_llama(args)