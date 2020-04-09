#!/bin/bash

# This script trains an edsr model.

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

echo "training with scale $scale"
python train.py \
    --model edsr \
    --name edsr_x${scale} \
    --scale $scale