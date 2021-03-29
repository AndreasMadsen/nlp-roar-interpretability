#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 380
source "batch_jobs/_job_script.sh"

# Actual   pre_time=( ["random"]="0:02:0" ["attention"]="0:02:0" ["gradient"]="0:02:0" ["integrated-gradient"]="0:02:0" )
declare -A pre_time=( ["random"]="0:15:0" ["attention"]="0:15:0" ["gradient"]="0:15:0" ["integrated-gradient"]="0:15:0" )

# Actual   roar_time="0:02:0"
declare -r roar_time="0:02:0"

for seed in {0..4}
do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
    do
        if precompute_jobid=$(
            sbatch --time=${pre_time[$importance_measure]} --mem=6G --parsable \
                -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                -J "sst-pre_s-${seed}_m-${importance_measure::1}_r-0_rs-50" $(job_script gpu) \
                experiments/compute_importance_measure.py \
                --seed ${seed} \
                --dataset sst \
                --importance-measure ${importance_measure} \
                --importance-caching build
        );  then
            echo "Submitted precompute batch job ${precompute_jobid}"
        else
            echo "Could not submit precompute batch job, skipping"
            break
        fi

        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/sst_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-50.json" ]; then
                echo sst_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-50
                sbatch --time=${roar_time} --mem=6G --dependency=afterok:${precompute_jobid} \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J sst_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-50 $(job_script gpu) \
                    experiments/stanford_sentiment.py \
                    --seed ${seed} --k ${k} --recursive-step-size 1 \
                    --roar-strategy count --importance-measure ${importance_measure} \
                    --importance-caching use
            fi
        done

        for k in {10..90..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/roar/sst_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-50.json" ]; then
                echo sst_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-50
                sbatch --time=${roar_time} --mem=6G --dependency=afterok:${precompute_jobid} \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J sst_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-50 $(job_script gpu) \
                    experiments/stanford_sentiment.py \
                    --seed ${seed} --k ${k} --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure ${importance_measure} \
                    --importance-caching use
            fi
        done
    done
done
