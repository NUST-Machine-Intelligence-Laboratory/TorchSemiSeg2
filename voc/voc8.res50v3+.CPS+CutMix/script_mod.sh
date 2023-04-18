#!/usr/bin/env bash
# nvidia-smi
export volna="/farm/chenrui/code/TorchSemiSeg2"
export NGPUS=1
export OUTPUT_PATH="/farm/chenrui/code/TorchSemiSeg2/A"
export snapshot_dir=$OUTPUT_PATH/snapshot
export log_dir=$OUTPUT_PATH/log
export labeled_ratio=16
export nepochs=32
export batch_size=6
export learning_rate=0.0025
export snapshot_iter=1
CUDA_VISIBLE_DEVICES=0 python train_mod.py
CUDA_VISIBLE_DEVICES=0 python eval_color.py -e 20-34 --save_path $OUTPUT_PATH/results
