#!/bin/bash

for seed in {0..4}
do
    for importance_measure in 'attention' 'gradient' 'integrated-gradient'
    do
        sbatch --time=2:00:0 --mem=32G \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J "importance_s-${seed}_m-${importance_measure::1}" ./python_job.sh \
            experiments/attention_distributions.py \
            --seed ${seed} \
            --importance-measure ${importance_measure}
    done
done
