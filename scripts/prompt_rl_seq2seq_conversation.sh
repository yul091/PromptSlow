
CUDA_VISIBLE_DEVICES=4 python prompt_transformer.py \
    --task conversational \
    --my_model_name t5-base \
    --temperature 0 \
    --saving_step 300 \
    --num_epochs 30 \
    --use_greedy_baseline


# --n_train_samples 100 \
# --n_test_samples 100 \