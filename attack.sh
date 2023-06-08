ATT_METHOD=prompt # prompt, fd, hotflip, textbugger, uat
MAX_PER=5
MODEL_PATH=microsoft/DialoGPT-medium # microsoft/DialoGPT-large, microsoft/DialoGPT-medium, microsoft/DialoGPT-small, facebook/blenderbot-400M-distill, facebook/blenderbot-1B-distill
DATASET=blended_skill_talk # blended_skill_talk, conv_ai_2, empathetic_dialogues, AlekseyKorshuk/persona-chat
FITNESS=length # performance, length, combined, adaptive
NUM_SAMPLES=30
MAX_LENGTH=256
SELECT_BEAMS=2

CUDA_VISIBLE_DEVICES=1 python -W ignore attack.py \
    --attack_strategy $ATT_METHOD \
    --max_per $MAX_PER \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --max_num_samples $NUM_SAMPLES \
    --max_len $MAX_LENGTH \
    --select_beams $SELECT_BEAMS \
    --out_dir logging \
    --fitness $FITNESS
    # --use_combined_loss