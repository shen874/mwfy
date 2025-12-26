#!/usr/bin/env bash
set -e
DATA_DIR=${1:-"./data"}
python -u src/train_braille.py --data_dir "$DATA_DIR" --epochs 15 --batch_size 128 --d_model 384 --nhead 6 --num_layers 4 --ffn_dim 1024 --dropout 0.1 --lr 3e-4 --max_len 1024
