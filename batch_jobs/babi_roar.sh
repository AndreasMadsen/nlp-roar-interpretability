#!/bin/bash
# jobs: 5 * 3 * 4 * (10 + 9) = 1140

declare -A time=( ["1 random"]="0:20:0" ["1 attention"]="0:20:0" ["1 gradient"]="0:20:0"
                  ["2 random"]="0:25:0" ["2 attention"]="0:25:0" ["2 gradient"]="0:25:0"
                  ["3 random"]="0:35:0" ["3 attention"]="0:40:0" ["3 gradient"]="0:40:0")

for seed in {0..4}
do
    for type in 1 2 3
    do
        for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
        do
            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0.json" ]; then
                    echo babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$type $importance_measure]} --mem=24G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/babi.py \
                        --seed ${seed} --k ${k} --recursive-step-size 1 \
                        --roar-strategy count --importance-measure ${importance_measure} \
                        --task ${type}
                fi
            done

            for k in {10..90..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                    echo babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$type $importance_measure]} --mem=24G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/babi.py \
                        --seed ${seed} --k ${k} --recursive-step-size 10 \
                        --roar-strategy quantile --importance-measure ${importance_measure} \
                        --task ${type}
                fi
            done
        done
    done
done
