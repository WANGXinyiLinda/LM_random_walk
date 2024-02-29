DATASET=gsm8k
MODE=supervised
MODEL=llama-2
EMBEDDING=llama-2
EFFICIENT=lora
FLASH=True
LR=2e-4
EPOCH=5
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path your_checkpoint \
    --base_model_name_or_path $MODEL \
    --parameter_efficient_mode $EFFICIENT \
    --mode $MODE \
    --dataset $DATASET \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/$MODEL/$DATASET/sft+random_walk-efficient=$EFFICIENT-lr=$LR-flash_attn=$FLASH-embed=$EMBEDDING\
    --model_max_length 512 \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 200 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --optim "adamw_torch" \
    --gradient_accumulation_steps 4 \
    --embedding_model_name $MODEL \
    --num_test 1000 \
    --lora_module mlp \
    --save_safetensors True \
    --int8_training True \
    --flash_attention $FLASH \
    --continue_training True \
    # --gradient_checkpointing \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --sharded_ddp "zero_dp_2 offload" \
    # --fsdp "full_shard offload" \