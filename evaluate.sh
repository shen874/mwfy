#!/usr/bin/env bash
set -e
DATA_DIR=${1:-"./data"}
python -u src/train_braille.py --data_dir "$DATA_DIR" --evaluate
