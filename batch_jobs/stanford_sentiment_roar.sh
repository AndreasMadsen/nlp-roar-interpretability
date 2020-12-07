#!/bin/bash
for seed in {0..4}
do
    for k in {1..5}
    do
        for masking in 'random' 'top-k'
        do
            sbatch --time=0:15:0 --mem=12G -J sst_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                experiments/stanford_sentiment_roar.py \
                --seed ${seed} --k ${k} --masking ${masking}
        done
    done
done
