#!/bin/bash
for seed in {0..4}
do
    for k in {1..5}
    do
        for masking in 'random' 'top-k'
        do
            sbatch --time=1:30:0 --mem=16G -J nli_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                experiments/stanford_nli.py \
                --seed ${seed} --k ${k} --masking ${masking}
        done
    done
done
