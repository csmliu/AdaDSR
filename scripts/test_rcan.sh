#!/bin/bash

# This script can generate exactly the same results with the official RCAN code,
# which can be found at https://github.com/yulunzhang/RCAN

# Reference:
# Zhang Y, Li K, Li K, et al. Image super-resolution using very deep residual
# channel attention networks[C]//Proceedings of the European Conference on
# Computer Vision (ECCV). 2018: 286-301.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model rcan \
    --name rcan_x${scale} \
    --scale $scale \
    --load_path ./pretrained/RCAN_BIX${scale}.pth \
    --dataset_name set5 \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0