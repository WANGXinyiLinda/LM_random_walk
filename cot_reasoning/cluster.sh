CUDA_VISIBLE_DEVICES=7 python cluster_steps.py \
    --model_name_or_path llama-2 \
    --batch_size 8 \
    --dataset gsm8k \
    --num_types 200 \