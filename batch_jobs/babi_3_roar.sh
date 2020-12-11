#!/bin/bash
for seed in {0..4}
do
    for k in {1..10}
    do
        for masking in 'random' 'top-k'
        do
            if [ ! -f $SCRATCH"/comp550/results/babi_t-3_roar_s-${seed}_k-${k}_m-${masking::1}.json" ]; then
                echo babi_t-3_roar_s-${seed}_k-${k}_m-${masking::1}
                sbatch --time=0:50:0 --mem=12G -J babi_t-3_roar_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                    experiments/babi_roar.py \
                    --seed ${seed} --k ${k} --masking ${masking} --task 3
            fi
        done
    done
done
