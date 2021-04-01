#!/bin/bash
source "batch_jobs/_job_script.sh"

# Actual time: "0:01:30"

for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/roar/sst_s-${seed}.json" ]; then
        echo sst_s-${seed}
        sbatch --time=0:10:0 --mem=6G \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J sst_s-${seed} $(job_script gpu) \
            experiments/stanford_sentiment.py \
            --seed ${seed}
    fi
done
