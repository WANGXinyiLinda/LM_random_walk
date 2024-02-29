   DATASET=umls
   CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name_or_path your_checkpoint \
    --data_dir data \
    --dataset $DATASET \
    --split test \