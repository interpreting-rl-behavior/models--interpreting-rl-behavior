#!/bin/bash
set -o errexit

num_seeds=10000
#num_seeds=10

run(){
    for random_percent in `seq 0 2 20`
    do
	echo running model trained on random_percent $random_percent
        python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent 100
    done
}


run_iid(){
    for random_percent in `seq 0 2 20`
    do
	echo running model trained on random_percent $random_percent
        python run_coinrun.py --model_file $random_percent --start_level_seed 0 --num_seeds $num_seeds --random_percent $random_percent --reset_mode "complete"
    done
}

run
run_iid

# for dummy in {1..10}
# do
#     run &
# done



