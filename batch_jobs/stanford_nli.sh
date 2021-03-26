#!/bin/bash
source "batch_jobs/_job_script.sh"

# Actual time: "0:44:0"
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}.json" ]; then
        echo snli_s-${seed}
        sbatch --time=1:10:0 --mem=24G -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J snli_s-${seed} $(job_script gpu) \
            experiments/stanford_nli.py \
            --seed ${seed}
    fi
done
