
# CUDA_VISIBLE_DEVICES='0' nohup python get_summarization_output.py \
#  --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
#  --prompt 'Summarize the following article into one sentence:\n' \
#  --dataset cnn \
#  --save_path ./cnn_train_one_vicuna.json > ./train_one.log 2>&1 &

# CUDA_VISIBLE_DEVICES='6' nohup python get_summarization_output.py \
#  --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
#  --prompt 'Summarize the following article into two sentences:\n' \
#  --dataset cnn \
#  --save_path ./cnn_train_two_vicuna.json > ./train_two.log 2>&1 &
 
# CUDA_VISIBLE_DEVICES='7' nohup python get_summarization_output.py \
#  --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
#  --prompt 'Summarize the following article into three sentences:\n' \
#  --dataset cnn \
#  --save_path ./cnn_train_three_vicuna.json > ./train_three.log 2>&1 &
 
CUDA_VISIBLE_DEVICES='0' nohup python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Answer the following instruct within one sentence:\n' \
 --dataset sharegpt \
 --save_path ./sharegpt_train_one_vicuna.json > ./train_one.log 2>&1 &

CUDA_VISIBLE_DEVICES='1' nohup python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Answer the following instruct within two sentences:\n' \
 --dataset sharegpt \
 --save_path ./sharegpt_train_two_vicuna.json > ./train_two.log 2>&1 &
 
CUDA_VISIBLE_DEVICES='6' nohup python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Answer the following instruct within three sentences:\n' \
 --dataset sharegpt \
 --save_path ./sharegpt_train_three_vicuna.json > ./train_three.log 2>&1 &

CUDA_VISIBLE_DEVICES='7' nohup python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Answer the following instruct within four sentences:\n' \
 --dataset sharegpt \
 --save_path ./sharegpt_train_four_vicuna.json > ./train_four.log 2>&1 &
 
# python assess_reward.py \
#  --model /data/yfu093/pre_model_save/llama-2-7b-chat-hf \