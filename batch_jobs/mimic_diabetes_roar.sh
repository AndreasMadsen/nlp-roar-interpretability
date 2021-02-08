#!/bin/bash
for seed in {0..4}
do
    for k in {1..10}
    do
        for masking in 'random' 'top-k'
        do
            if [ ! -f $SCRATCH"/comp550/results/mimic_t-d_s-${seed}_k-${k}_m-${masking::1}.json" ]; then
                echo mimic_t-d_s-${seed}_k-${k}_m-${masking::1}
                sbatch --time=0:40:0 --mem=16G -J mimic-t-d_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                    experiments/mimic.py \
                    --seed ${seed} --k ${k} --masking ${masking} \
                    --subset diabetes
            fi
        done
    done
done
