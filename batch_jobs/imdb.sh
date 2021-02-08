#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/imdb_s-${seed}_k-0_m-a_r-0.json" ]; then
        echo imdb_s-${seed}
        sbatch --time=0:15:0 --mem=12G \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J imdb_s-${seed}_k-0_m-a_r-0 ./python_job.sh \
            experiments/imdb.py \
            --seed ${seed}
    fi
done
