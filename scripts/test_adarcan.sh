#!/bin/bash
if [ -n "$1" ]; then
    scale=$1
else
    scale=2
fi

if [ -n "$2" ]; then
    depth=$2
else
    depth=20
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model adarcan \
    --name adarcan_x${scale} \
    --scale $scale \
    --load_path ./ckpt/adarcan_x${scale}/AdaRCAN_model.pth \
    --dataset_name set5 \
    --depth $depth \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0