#!/bin/bash

# This script trains a vdsr model.

# The input is image super-resolved by bicubic algorithm, and only Y channel
# (of YCbCr color space) is used.
# In the evaluation procedure (set `--calc_psnr True`), the Y channel output is
# directly used to calculate the PSNR index.

# Reference:
# Kim J, Kwon Lee J, Mu Lee K. Accurate image super-resolution using very deep
# convolutional networks[C]//Proceedings of the IEEE conference on computer
# vision and pattern recognition. 2016: 1646-1654.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "training with scale $scale"
python train.py \
    --model vdsr \
    --name vdsr_x${scale} \
    --scale $scale