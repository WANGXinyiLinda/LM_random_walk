DATASET=gsm8k
MODE=random_walk
MODEL=llama-2
EFFICIENT=lora
USE_DEMO=False
CLUSTER=k-means
NODES=10
COT=10
CONTI=5
EMBEDDING=llama-2
K_DEMO=4
LR=2e-4
FLASH=True
CUDA_VISIBLE_DEVICES=7 python train.py \
    --model_name_or_path $MODEL \
    --parameter_efficient_mode $EFFICIENT \
    --mode $MODE \
    --dataset $DATASET \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/$MODEL/$DATASET/random_walk-efficient=$EFFICIENT-lr=$LR-cluster=$CLUSTER-nodes=$NODES-embed=$EMBEDDING-cot=$COT-conti=$CONTI-flash_attn=$FLASH \
    --model_max_length 1024 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 200 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --optim "adamw_torch" \
    --gradient_accumulation_steps 8 \
    --cluster_model_name $CLUSTER \
    --num_nodes $NODES \
    --embedding_model_name $EMBEDDING \
    --cot_length $COT \
    --continuous_steps $CONTI \
    --lora_module mlp \
    --save_safetensors True \
    --int8_training True \
    --flash_attention $FLASH \
    # --gradient_checkpointing \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --sharded_ddp "zero_dp_2 offload" \
    # --fsdp "full_shard offload" \