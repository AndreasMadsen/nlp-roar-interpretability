#!/bin/bash
for seed in {0..4}
do
    for type in 1 2 3
    do
        if [ ! -f $SCRATCH"/comp550/results/babi_t-${type}_s-${seed}_k-0_m-a_r-0.json" ]; then
            echo babi_t-${type}_s-${seed}
            sbatch --time=0:45:0 --mem=12G \
                -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                -J babi_t-${type}_s-${seed}_k-0_m-a_r-0 ./python_job.sh \
                experiments/babi.py \
                --seed ${seed} \
                --task ${type}
        fi
    done
done
