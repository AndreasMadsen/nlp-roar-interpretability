#!/bin/bash
# Actual time: "0:04:0"
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/roar/imdb_s-${seed}.json" ]; then
        echo imdb_s-${seed}
        sbatch --time=0:15:0 --mem=6G \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J imdb_s-${seed} ./python_job.sh \
            experiments/imdb.py \
            --seed ${seed}
    fi
done
