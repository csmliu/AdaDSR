#!/bin/bash

# This script generates !! nearly !! the same results with the official code,
# which can be found at https://github.com/yulunzhang/RDN (a torch version), as
# we converted the official torch models to pytorch models.
# If exactly official results are required, please refer to the authors' repo.

# Reference:
# Zhang Y, Tian Y, Kong Y, et al. Residual dense network for image
# super-resolution[C]//Proceedings of the IEEE conference on computer vision
# and pattern recognition. 2018: 2472-2481.

if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model rdn \
    --name rdn_x${scale} \
    --scale $scale \
    --load_path ./ckpt/rdn_x${scale}/RDN_BIX${scale}.pth \
    --dataset_name set5 \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0