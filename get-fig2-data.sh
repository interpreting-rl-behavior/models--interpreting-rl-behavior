#!/bin/bash
set -o errexit

num_seeds=10000

for dummy in {1..10}
do
    for random_percent in 0 1 2 3 6 11
    do
        python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent 100
    done
done
