#!/bin/sh

dataset="pacnerf"
scene_list="torus bird trophy playdoh cat cream toothpaste droplet letter"

for s in $scene_list; do
    if [ "$s" = "cream" ]; then
        n_frame=16
    elif [ "$s" = "toothpaste" ]; then
        n_frame=10
    elif [ "$s" = "letter" ]; then
        n_frame=15
    else
        n_frame=13
    fi

    python simulation.py --model_path "./model/${dataset}/${s}" \
    --n_epoch 50 --output_path "./outputs/${dataset}/${s}" \
    --physics_config "./config/${dataset}/${s}_config.json" --n_key_frame ${n_frame} # --eval

    ### uncomment --eval for evaluation after training
done