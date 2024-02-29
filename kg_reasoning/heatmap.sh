DATASET=countries_S3
CUDA_VISIBLE_DEVICES=0 python analysis.py \
    --data_dir data \
    --dataset $DATASET \
    --split test \
    --model_name_or_path your_checkpoint \
    --pra_temp 100 \
    --restricted_vocab all_entities \
    --load_in_16bits True \
    --max_rule_len 3 \