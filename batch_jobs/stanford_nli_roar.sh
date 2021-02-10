#!/bin/bash
for seed in {0..4}
do
    for importance_measure in 'random' 'attention' 'gradient'
    do
        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}_k-${k}_m-${importance_measure::1}_r-0.json" ]; then
                echo snli_s-${seed}_k-${k}_m-${importance_measure::1}_r-0
                sbatch --time=2:30:0 --mem=32G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_s-${seed}_k-${k}_m-${importance_measure::1}_r-0 ./python_job.sh \
                    experiments/stanford_nli.py \
                    --seed ${seed} --k ${k} --importance-measure ${importance_measure}
            fi
        done
    done
done
