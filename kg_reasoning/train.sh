DATASET=umls
MODE=random_walk
PATH_LEN=5
LR=5e-4
BATCH=16
MAX_LEN=1024
MODEL=gpt2
CUDA_VISIBLE_DEVICES=7 python train.py \
    --model_type Transformer \
    --model_name_or_path $MODEL \
    --random_initialize True \
    --mode $MODE \
    --path_len $PATH_LEN \
    --data_dir data \
    --dataset $DATASET \
    --fp16 True \
    --output_dir checkpoints/$DATASET/pretrain-$MODE-triple-$MODEL-len=$PATH_LEN-lr=$LR-batch=$BATCH-$MAX_LEN \
    --model_max_length $MAX_LEN \
    --max_steps 30000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 200 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --entity_as_new_token True \
    --relation_as_new_token True \
    # --resume True \