#!/bin/bash

# This script trains an SAN model.

# NOTE that currently supports single-GPU training only.

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

echo "training with scale $scale"
python test.py \
    --model san \
    --name san_x${scale} \
    --scale $scale \
    --gpu_ids 0 \
    --chop True # otherwise, may cause `Out Of Memory (OOM)` error