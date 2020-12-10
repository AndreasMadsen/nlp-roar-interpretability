#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/babi_t-1_s-${seed}.json" ]; then
        echo babi_t-1_s-${seed}
        sbatch --time=0:20:0 --mem=12G -J babi_t-1_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err"  ./python_job.sh \
            experiments/babi.py \
            --seed ${seed} \
            --task 1
    fi
done
