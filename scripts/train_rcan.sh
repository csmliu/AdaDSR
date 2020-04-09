#!/bin/bash

# This script trains a rcan model.

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

echo "training with scale $scale"
python test.py \
    --model rcan \
    --name rcan_x${scale} \
    --scale $scale