#!/bin/bash
source "batch_jobs/_job_script.sh"

for seed in {0..4}
do
    for importance_measure in 'attention' 'gradient' 'integrated-gradient'
    do
        sbatch --time=2:00:0 --mem=16G \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J "importance_s-${seed}_m-${importance_measure::1}" $(job_script gpu) \
            experiments/analyze_importance_measure.py \
            --seed ${seed} \
            --importance-measure ${importance_measure}
    done
done
