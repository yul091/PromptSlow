
#! cnn
CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_none_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into one sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_one_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into two sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_two_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into three sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_three_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into four sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_four_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into five sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_five_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into six sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_six_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into seven sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_seven_vicuna.json

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into eight sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_eight_vicuna.json 

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into nine sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_nine_vicuna.json 

CUDA_VISIBLE_DEVICES='0' python get_summarization_output.py \
 --model /data/yfu093/pre_model_save/vicuna-7b-v1.3 \
 --prompt 'Summarize the following article into ten sentences:\n' \
 --dataset cnn \
 --save_path ./result/cnn_dailymail/cnn_train_ten_vicuna.json 