
CUDA_VISIBLE_DEVICES=0 python prompt_transformer.py \
    --task conversational \
    --my_model_name facebook/blenderbot-400M-distill \
    --temperature 0 \
    --n_train_samples 10 \
    --n_test_samples 10 \
    --saving_step 300 \
    --num_epochs 30 \
    --use_greedy_baseline


# --n_train_samples 100 \
# --n_test_samples 100 \