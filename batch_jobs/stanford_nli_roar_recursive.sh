#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 285

# Actual time:    ["random"]="0:51:0" ["attention"]="0:46:0" ["gradient"]="0:48:0" ["integrated-gradient"]="0:55:0"
declare -A time=( ["random"]="1:15:0" ["attention"]="1:15:0" ["gradient"]="1:15:0" ["integrated-gradient"]="1:25:0" )

for seed in {0..4}
do
    for importance_measure in 'attention' 'gradient' 'integrated-gradient'
    do
        dependency=''

        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1.json" ]; then
                echo snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1
                if last_jobid=$(
                    sbatch --time=${time[$importance_measure]} --mem=24G --parsable ${dependency} \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1 ./python_job.sh \
                    experiments/stanford_nli.py --recursive \
                    --seed ${seed} --k ${k} --recursive-step-size 1 \
                    --roar-strategy count --importance-measure ${importance_measure}
                ); then
                    echo "Submitted batch job ${last_jobid}"
                    dependency="--dependency=afterok:${last_jobid}"
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            fi
        done

        dependency=''

        for k in {10..90..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1.json" ]; then
                echo snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1
                if last_jobid=$(
                    sbatch --time=${time[$importance_measure]} --mem=24G --parsable ${dependency} \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1 ./python_job.sh \
                    experiments/stanford_nli.py --recursive \
                    --seed ${seed} --k ${k} --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure ${importance_measure}
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
