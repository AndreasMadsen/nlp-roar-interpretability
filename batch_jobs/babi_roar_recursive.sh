#!/bin/bash
# jobs: 5 * 3 * 2 * (10 + 9) = 570

declare -A time=( ["1 random"]="0:20:0" ["1 attention"]="0:20:0" ["1 gradient"]="0:20:0"
                  ["2 random"]="0:40:0" ["2 attention"]="0:40:0" ["2 gradient"]="0:40:0"
                  ["3 random"]="0:50:0" ["3 attention"]="0:50:0" ["3 gradient"]="0:50:0")

for seed in {0..4}
do
    for type in 1 2 3
    do
        for importance_measure in 'attention' 'gradient'
        do
            dependency=''

            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1.json" ]; then
                    echo babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1
                    if last_jobid=$(
                        sbatch --time=${time[$type $importance_measure]} --mem=24G --parsable ${dependency} \
                            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                            -J babi-${type}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1 ./python_job.sh \
                            experiments/babi.py --recursive \
                            --seed ${seed} --k ${k} --recursive-step-size 1 \
                            --roar-strategy count --importance-measure ${importance_measure} \
                            --task ${type}
                    ); then
                        echo "Submitted batch job ${last_jobid}"
                        dependency="--dependency=afterok:${last_jobid}"
                    else
                        echo "Could not submit batch job, skipping"
                        break
                    fi
                fi
            done

            for k in {10..90..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1.json" ]; then
                    echo babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1
                    if last_jobid=$(
                        sbatch --time=${time[$type $importance_measure]} --mem=24G --parsable ${dependency} \
                            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                            -J babi-${type}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1 ./python_job.sh \
                            experiments/babi.py --recursive \
                            --seed ${seed} --k ${k} --recursive-step-size 10 \
                            --roar-strategy quantile --importance-measure ${importance_measure} \
                            --task ${type}
                    ); then
                        echo "Submitted batch job ${last_jobid}"
                        dependency="--dependency=afterok:${last_jobid}"
                    else
                        echo "Could not submit batch job, skipping"
                        break
                    fi
                fi
            done
        done
    done
done
