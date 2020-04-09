#!/bin/bash
if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

cd ..

echo "training with scale $scale"
python train.py \
    --model adaedsr \
    --name adaedsr_x${scale} \
    --scale $scale \
    --load_path ./pretrained/EDSR_official_32_x${scale}.pth