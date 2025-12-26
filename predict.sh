#!/usr/bin/env bash
set -e
DATA_DIR=${1:-"./data"}
TEXT=${2:-"你好，世界！"}
python -u src/train_braille.py --data_dir "$DATA_DIR" --predict "$TEXT"
