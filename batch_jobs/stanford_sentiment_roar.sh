#!/bin/bash
for seed in {0..4}
do
    for importance_measure in 'random' 'attention'
    do
        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/sst_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-0.json" ]; then
                echo sst_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-0
                sbatch --time=0:15:0 --mem=12G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J sst_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-0 ./python_job.sh \
                    experiments/stanford_sentiment.py \
                    --seed ${seed} --k ${k} --importance-measure ${importance_measure}
            fi
        done
    done
done
