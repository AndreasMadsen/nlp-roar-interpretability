#!/bin/bash
for seed in {0..4}
do
    for k in {0..4}
    do
        for random_masking in {0..1}
        do
            sbatch --time=0:30:0 -J sst_s-${seed}_k-${k}_r-${random_masking} -o ${SCRATCH}"/logs/slurm-%j.err" -e ${SCRATCH}"/logs/slurm-%j.err" ./python_job.sh \
                experiments/stanford_sentiment_roar.py \
                --seed ${seed} --k ${k} --random-masking ${random_masking}
        done
    done
done
