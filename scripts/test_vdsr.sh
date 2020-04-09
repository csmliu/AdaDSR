#!/bin/bash

# This script is only used for counting inference time and calculating FLOPs,
# and the given checkpoint file is not able to generate official results.
# To get qualitative results and PSNR/SSIM indices, please refer to the authors'
# project: https://cv.snu.ac.kr/research/VDSR/

# Reference:
# Kim J, Kwon Lee J, Mu Lee K. Accurate image super-resolution using very deep
# convolutional networks[C]//Proceedings of the IEEE conference on computer
# vision and pattern recognition. 2016: 1646-1654.

# Note that VDSR takes super-resolved image as input.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model vdsr \
    --name vdsr_x${scale} \
    --scale $scale \
    --dataset_name set5 \
    --load_path ./ckpt/vdsr/vdsr_model.pth \
    --matlab True \
    --sparse_conv True \
    --gpu_ids 0