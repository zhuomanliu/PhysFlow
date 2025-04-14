#!/bin/sh

dataset="phys_dreamer"
scene_list="alocasia carnation hat telephone"


for s in $scene_list; do

    python simulation.py --model_path ./model/${dataset}/${s} \
    --n_epoch 10 --output_path ./outputs/${dataset}/${s} \
    --physics_config ./config/${dataset}/${s}_config.json # --eval

    ### uncomment --eval for evaluation after training
done