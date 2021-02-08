#!/bin/bash
for seed in {0..4}
do
    for k in {1..10}
    do
        for masking in 'random' 'top-k'
        do
            if [ ! -f $SCRATCH"/comp550/results/mimic_t-a_s-${seed}_k-${k}_m-${masking::1}.json" ]; then
                echo mimic_t-a_s-${seed}_k-${k}_m-${masking::1}
                sbatch --time=0:30:0 --mem=16G -J mimic-t-a_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                    experiments/mimic.py \
                    --seed ${seed} --k ${k} --masking ${masking} \
                    --subset anemia
            fi
        done
    done
done
