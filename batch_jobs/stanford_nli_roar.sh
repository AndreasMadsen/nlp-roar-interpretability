#!/bin/bash
for seed in {0..4}
do
    for k in {1..14}
    do
        for masking in 'random' 'top-k'
        do
            if [ ! -f $SCRATCH"/comp550/results/snli_roar_s-${seed}_k-${k}_m-${masking::1}.json" ]; then
                echo snli_roar_s-${seed}_k-${k}_m-${masking::1}
                sbatch --time=1:30:0 --mem=32G -J snli_roar_s-${seed}_k-${k}_m-${masking::1} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
                    experiments/stanford_nli_roar.py \
                    --seed ${seed} --k ${k} --masking ${masking}
            fi
        done
    done
done
