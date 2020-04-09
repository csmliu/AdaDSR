#!/bin/bash

# This script is only used for counting inference time and calculating FLOPs,
# and the given checkpoint file is converted from the authors' pytorch version,
# which is slightly higher than their torch version (used in their paper).
# See https://github.com/thstkdgus35/EDSR-PyTorch for official pytorch version.
# To get qualitative results and PSNR/SSIM indices, please refer to the authors'
# torch version: https://github.com/LimBee/NTIRE2017

# Reference:
# Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image
# super-resolution[C]//Proceedings of the IEEE conference on computer vision and
# pattern recognition workshops. 2017: 136-144.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model edsr \
    --name edsr_x${scale} \
    --scale $scale \
    --load_path ./pretrained/EDSR_official_32_x${scale}.pth \
    --dataset_name set5 \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0