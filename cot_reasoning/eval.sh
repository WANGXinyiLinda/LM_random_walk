DATASET=aqua
MODEL=llama-2
EFFICIENT=lora
FLASH=True
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --base_model_name_or_path $MODEL \
    --model_name_or_path your_checkpoint \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 8 \
    --max_length 1024 \
    --seed 100 \
    --load_in_8bit True \
    --flash_attention $FLASH \
    # --num_test 1000 \
    # --prompt_template alpaca \
    # --use_calculator True \