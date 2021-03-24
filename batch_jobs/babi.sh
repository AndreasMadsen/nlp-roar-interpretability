#!/bin/bash
# Actual time:    ["1"]="0:07:0" ["2"]="0:10:0" ["3"]="0:22:0"
declare -A time=( ["1"]="0:20:0" ["2"]="0:30:0" ["3"]="0:40:0")

for seed in {0..4}
do
    for type in 1 2 3
    do
        if [ ! -f $SCRATCH"/comp550/results/roar/babi-${type}_s-${seed}.json" ]; then
            echo babi-${type}_s-${seed}
            sbatch --time=${time[$type]} --mem=6G \
                -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                -J babi-${type}_s-${seed} ./python_job.sh \
                experiments/babi.py \
                --seed ${seed} \
                --task ${type}
        fi
    done
done
