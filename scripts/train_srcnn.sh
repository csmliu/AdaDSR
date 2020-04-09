#!/bin/bash

# This script trains a srcnn model.

# The input is image super-resolved by bicubic algorithm, and only Y channel
# (of YCbCr color space) is used.
# In the evaluation procedure (set `--calc_psnr True`), the Y channel output is
# directly used to calculate the PSNR index.

# Reference:
# Dong C, Loy C C, He K, et al. Image super-resolution using deep convolutional
# networks[J]. IEEE transactions on pattern analysis and machine intelligence,
# 2015, 38(2): 295-307.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "training with scale $scale"
python train.py \
    --model srcnn \
    --name srcnn_x${scale} \
    --scale $scale