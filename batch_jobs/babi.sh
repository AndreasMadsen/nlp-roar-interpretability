#!/bin/bash
declare -A time=( ["1"]="0:20:0" ["2"]="0:40:0" ["3"]="0:50:0")

for seed in {0..4}
do
    for type in 1 2 3
    do
        if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}.json" ]; then
            echo babi-${type}_s-${seed}
            sbatch --time=${time[$type]} --mem=24G \
                -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                -J babi-${type}_s-${seed} ./python_job.sh \
                experiments/babi.py \
                --seed ${seed} \
                --task ${type}
        fi
    done
done
