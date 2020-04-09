#!/bin/bash

# This script is only used for counting inference time and calculating FLOPs.
# To get qualitative results and PSNR/SSIM indices, please refer to the authors'
# project: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html

# Reference:
# Dong C, Loy C C, He K, et al. Image super-resolution using deep convolutional
# networks[J]. IEEE transactions on pattern analysis and machine intelligence,
# 2015, 38(2): 295-307.

# Note that SRCNN takes super-resolved image as input.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model srcnn \
    --name srcnn_${scale} \
    --scale $scale \
    --dataset_name set5 \
    --gpu_ids 0