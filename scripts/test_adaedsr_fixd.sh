#!/bin/bash
# This script is for `FAdaEDSR` in the ablation study of the paper.

scale=2
# Only models of scale x2 are provided, you can train with train_adaedsr_fixd.sh
# if other conditions are needed.
# Note that for convenience, `--depth` is set to 1 by default in all conditions,
# which is equivalent to removing the desired depth $d$. Though 'depth' is used
# as a parameter here, it means the desired depth $d$ in the training procedure.

if [ -n "$1" ]; then
    depth=$1
else
    depth=32
fi

cd ..

echo "testing with scale $scale"
python test.py \
    --model adaedsr_fixd \
    --name adaedsr_fixd_32_x2_d${depth} \
    --scale $scale \
    --load_path ./ckpt/adaedsr_fixd_32_x2_d${depth}/AdaEDSRFixD_model.pth \
    --dataset_name set5 \
    --chop True \
    --sparse_conv True \
    --matlab True \
    --gpu_ids 0