#!/bin/bash

# This script trains a rdn model.

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

echo "training with scale $scale"
python test.py \
    --model rdn \
    --name rdn_x${scale} \
    --scale $scale