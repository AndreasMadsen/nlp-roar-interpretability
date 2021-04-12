#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 380
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   pre_time=( ["random"]="0:02:0" ["attention"]="0:02:0" ["gradient"]="0:02:0" ["integrated-gradient"]="0:02:0" )
declare -A pre_time=( ["random"]="0:15:0" ["attention"]="0:15:0" ["gradient"]="0:15:0" ["integrated-gradient"]="0:15:0" )

# Actual   roar_time="0:02:0"
declare -r roar_time="0:02:0"

for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
do
    riemann_samples=$(( $importance_measure == integrated-gradient ? 50 : 0 ))

    if precompute_jobid=$(
        submit_seeds ${pre_time[$importance_measure]} "$seeds" "importance-measure/sst-pre_s-%s_m-${importance_measure::1}_rs-${riemann_samples}.csv.gz" \
            --mem=6G --parsable \
            $(job_script gpu) \
            experiments/compute_importance_measure.py \
            --dataset sst \
            --importance-measure "$importance_measure" \
            --importance-caching build
    );  then
        echo "Submitted precompute batch job $precompute_jobid"
    else
        echo "Could not submit precompute batch job, skipping"
        break
    fi

    for k in {1..10}
    do
        submit_seeds ${roar_time} "$seeds" "roar/sst_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
            --mem=6G --dependency=afterok:"$precompute_jobid" \
            $(job_script gpu) \
            experiments/stanford_sentiment.py \
            --k "$k" --recursive-step-size 1 \
            --roar-strategy count --importance-measure "$importance_measure" \
            --importance-caching use
    done

    for k in {10..90..10}
    do
        submit_seeds ${roar_time} "$seeds" "roar/sst_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
            --mem=6G --dependency=afterok:"$precompute_jobid" \
            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            $(job_script gpu) \
            experiments/stanford_sentiment.py \
            --k "$k" --recursive-step-size 10 \
            --roar-strategy quantile --importance-measure "$importance_measure" \
            --importance-caching use
    done
done
