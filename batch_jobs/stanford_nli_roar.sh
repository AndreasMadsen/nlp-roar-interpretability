#!/bin/bash
for seed in {0..0}
do
    for k in {1..2}
    do
        for masking in 'random' 'top-k'
        do
            sbatch --time=1:30:0 --mem=32G -J nli_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                experiments/stanford_nli_roar.py \
                --seed ${seed} --k ${k} --masking ${masking}
        done
    done
done
