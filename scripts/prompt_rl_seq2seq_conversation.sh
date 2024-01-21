
CUDA_VISIBLE_DEVICES=0 python prompt_transformer.py \
    --task conversational \
    --temperature 0 \
    --n_train_samples 30 \
    --n_test_samples 30 \
    --saving_step 10 \
    --num_epochs 10