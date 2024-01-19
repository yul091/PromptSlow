import os
import re
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor()

# !wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
prompt_complex = open("./prompt_hardest.txt").read()
gsm8k = load_dataset("gsm8k", "main")
gsm8k_test = gsm8k["test"]

access_token = "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"
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=access_token)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token=access_token)
# model = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear},  # a set of layers to dynamically quantize
#     dtype=torch.qint8,
# ) # the target dtype for quantized weights
device = 'cuda:0'
model = model.to(device)

question = gsm8k['train'][0]['question']
answer = gsm8k['train'][0]['answer']
print("Question:", question)
# example = prompt_complex.split("\n\n")[0]
instruction = "Please reference the following examples to answer the math question,\n"
prompt = instruction + prompt_complex + "\n\nQuestion: " + question
# print("Prompt:", prompt)
inputs = tokenizer(prompt, return_tensors='pt').to(device)
# Greedy decoding with a temperature of 0 to improve stability
output = model.generate(**inputs)
print("Response:", tokenizer.decode(output[0], skip_special_tokens=True))
print("Answer:", answer)

compressed_prompt = llm_lingua.compress_prompt(
    prompt_complex.split("\n\n"),
    instruction="",
    question="",
    target_token=200,
    context_budget="*1.5",
    iterative_size=100,
)
prompt = instruction + compressed_prompt["compressed_prompt"] + "\n\nQuestion: " + question
print("Compressed prompt:", prompt)
inputs = tokenizer(prompt, return_tensors='pt').to(device)
# Greedy decoding with a temperature of 0 to improve stability
output = model.generate(**inputs)
print("New Response:", tokenizer.decode(output[0], skip_special_tokens=True))


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