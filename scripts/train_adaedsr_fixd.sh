#!/bin/bash
if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

if [ -n "$2" ]; then
    depth=$2
else
    depth=32
fi

cd ..

echo "training with scale $scale"
python train.py \
    --model adaedsr_fixd \
    --name adaedsr_fixd_32_x${scale}_d${depth} \
    --scale $scale \
    --depth $depth \
    --load_path ./pretrained/EDSR_official_32_x${scale}.pth