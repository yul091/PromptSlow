from fastchat.model import add_model_args, load_model, get_conversation_template
import argparse
import json
from typing import Union, Tuple, Any, List
from rouge import Rouge
from tqdm import tqdm
rouge = Rouge()

def cutwords(sens: List[str], max_num_of_chars: int) -> List[str]:
    output = []
    quota = max_num_of_chars
    for sen in sens:
        if quota > len(sen):
            output.append(sen)
            quota -= len(sen)
        else:
            output.append(sen[:quota])
            break
    return output

def RougeTest_rouge(
    ref: List[str], 
    hyp: List[str], 
    rouge_metric: str = "all", 
    max_num_of_bytes: int = -1,
) -> Union[float, Tuple]:
    ref = [_.lower() for _ in ref]
    hyp = [_.lower() for _ in hyp]

    if max_num_of_bytes > 0:
        ref = cutwords(ref)
        hyp = cutwords(hyp)

    rouge_score = rouge.get_scores(hyp, ref)
    if rouge_metric[1] == 'f':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['f']
    elif rouge_metric[1] == 'r':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['r']
    elif rouge_metric == 'avg_f':
        return (rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 3
    elif rouge_metric == 'avg_r':
        return (rouge_score[0]['rouge-1']['r'] + rouge_score[0]['rouge-2']['r'] + rouge_score[0]['rouge-l']['r']) / 3
    else:
        return (rouge_score[0]['rouge-1']['p'], rouge_score[0]['rouge-1']['r'], rouge_score[0]['rouge-1']['f'],
                rouge_score[0]['rouge-2']['p'], rouge_score[0]['rouge-2']['r'], rouge_score[0]['rouge-2']['f'],
                rouge_score[0]['rouge-l']['p'], rouge_score[0]['rouge-l']['r'], rouge_score[0]['rouge-l']['f'])

if __name__ == "__main__":
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
    
    # model, tokenizer = load_model(
    #     args.model_path,
    #     device=args.device,
    #     num_gpus=args.num_gpus,
    #     max_gpu_memory=args.max_gpu_memory,
    #     load_8bit=args.load_8bit,
    #     cpu_offloading=args.cpu_offloading,
    #     revision=args.revision,
    #     debug=args.debug,
    # )

    total_data_path = [
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_one.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_two.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_three.json'

        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_one_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_two_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_three_vicuna.json' 
        # '/data/yfu093/PromptSlow/summ_preprocess/billsum_train_none_vicuna.json' 

        # '/data/yfu093/PromptSlow/summ_preprocess/cnn_train_one_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/cnn_train_two_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/cnn_train_three_vicuna.json' 
        # '/data/yfu093/PromptSlow/summ_preprocess/cnn_train_none_vicuna.json' 

        # '/data/yfu093/PromptSlow/summ_preprocess/sharegpt_train_one_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/sharegpt_train_two_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/sharegpt_train_three_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/sharegpt_train_four_vicuna.json' 

        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_none_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_one_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_two_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_three_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_four_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_five_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_six_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_seven_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_eight_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_nine_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/vicuna/cnn_train_ten_vicuna.json'


        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_none_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_one_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_two_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_three_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_four_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_five_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_six_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_seven_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_eight_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_nine_llama2.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/cnn_dailymail/llama2/cnn_train_ten_llama2.json'

        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_none_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_one_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_two_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_three_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_four_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_five_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_six_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_seven_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_eight_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_nine_vicuna.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/vicuna/xsum_train_ten_vicuna.json',

        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_none_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_one_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_two_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_three_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_four_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_five_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_six_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_seven_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_eight_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_nine_llama2.json',
        '/data/yfu093/PromptSlow/summ_preprocess/result/xsum/llama2/xsum_train_ten_llama2.json',

        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_none_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_one_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_two_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_three_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_four_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_five_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_six_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_seven_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_eight_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_nine_vicuna.json',
        # '/data/yfu093/PromptSlow/summ_preprocess/result/pubmed/vicuna/pubmed_train_ten_vicuna.json',
    ]
    from datasets import load_metric
    bert_score = load_metric("bertscore")
    for data_path in total_data_path:
        total_length_reward = []
        total_quality_reward = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            total_ref = []
            total_result = []
            for item in tqdm(data, total=len(data)):
                ref = item['reference']
                result = item['output']
                if len(ref) == 0 or len(result) == 0:
                    continue
                else:
                    total_ref.append(ref)
                    total_result.append(result)
                    try:
                        length_reward = len(ref) / len(result)
                        quality_reward = RougeTest_rouge([ref], [result], rouge_metric='avg_f')

                        total_length_reward.append(length_reward)
                        total_quality_reward.append(quality_reward)
                    except:
                        continue
            # l = sum(total_length_reward) / len(total_length_reward)
            # q = sum(total_quality_reward) / len(total_quality_reward)
            # print(f'data: {data_path} \n length_reward: {l} \n quality_reward: {q}')
            # print('-'*50)
            bert_reward = bert_score.compute(predictions=total_result, references=total_ref, lang="en")
            l = sum(total_length_reward) / len(total_length_reward)
            q = sum(total_quality_reward) / len(total_quality_reward)
            bert_reward = sum(bert_reward['f1']) / len(bert_reward['f1'])
            print(f'data: {data_path} \n len: {len(total_quality_reward)}\n length_reward: {l} \n quality_reward: {q} \n bert reward: {bert_reward}')
            print('-'*50)

        