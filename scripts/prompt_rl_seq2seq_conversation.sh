
CUDA_VISIBLE_DEVICES=0 python prompt_transformer.py \
    --task conversational \
    --my_model_name t5-small \
    --temperature 0 \
    --n_train_samples 100 \
    --n_test_samples 100 \
    --saving_step 100 \
    --num_epochs 10 \
    --use_greedy_baseline