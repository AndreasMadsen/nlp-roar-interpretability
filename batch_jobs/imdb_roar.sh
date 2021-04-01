#!/bin/bash
# jobs: 5 * 1 * 4 * (10 + 9) = 380
source "batch_jobs/_job_script.sh"

declare -A time=( ["random"]="0:20:0" ["attention"]="0:20:0" ["gradient"]="0:20:0" ["integrated-gradient"]="0:25:0" )

for seed in {0..4}
do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
    do
        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/imdb_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0.json" ]; then
                echo imdb_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0
                sbatch --time=${time[$importance_measure]} --mem=6G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J imdb_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0 $(job_script gpu) \
                    experiments/imdb.py \
                    --seed ${seed} --k ${k} --recursive-step-size 1 \
                    --roar-strategy count --importance-measure ${importance_measure}
            fi
        done

        for k in {10..90..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/imdb_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                echo imdb_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                sbatch --time=${time[$importance_measure]} --mem=6G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J imdb_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 $(job_script gpu) \
                    experiments/imdb.py \
                        --seed ${seed} --k ${k} --recursive-step-size 10 \
                        --roar-strategy quantile --importance-measure ${importance_measure}
            fi
        done
    done
done
