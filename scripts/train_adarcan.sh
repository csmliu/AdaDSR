#!/bin/bash
if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "training with scale $scale"
python train.py \
    --model adarcan \
    --name adarcan_x${scale} \
    --scale $scale \
    --load ./pretrained/RCAN_BIX${scale}.pth