#!/bin/bash

# This script can generate exactly the same results with the official SAN code,
# which can be found at https://github.com/daitao/SAN

# NOTE that we optimized `class Covpool` (AdaDSR/models/MPNCOV/python/MPNCOV.py)
# for faster inference, and you may obtain much shorter inference time than that 
# reported in the paper using this script.

# Reference:
# Dai T, Cai J, Zhang Y, et al. Second-order attention network for single image
# super-resolution[C]//Proceedings of the IEEE Conference on Computer Vision and
# Pattern Recognition. 2019: 11065-11074.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model san \
    --name san_x${scale} \
    --scale $scale \
    --load_path ./ckpt/san_model/SAN_BIX${scale}.pth \
    --dataset_name set5 \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0