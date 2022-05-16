#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 380
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   pre_time=( ["rnn random"]="0:02:0"     ["rnn mutual-information"]="0:04:0"     ["rnn attention"]="0:02:0"     ["rnn gradient"]="0:02:0"     ["rnn integrated-gradient"]="0:02:0"     ["rnn times-input-gradient"]="0:??:0" )
#                     ["roberta random"]="0:02:0" ["roberta mutual-information"]="0:??:0"                                ["roberta gradient"]="0:??:0" ["roberta integrated-gradient"]="0:??:0" ["roberta times-input-gradient"]="0:??:0" )
declare -A pre_time=( ["rnn random"]="0:15:0"     ["rnn mutual-information"]="0:15:0"     ["rnn attention"]="0:15:0"     ["rnn gradient"]="0:15:0"     ["rnn integrated-gradient"]="0:15:0"     ["rnn times-input-gradient"]="0:15:0"
                      ["roberta random"]="0:15:0" ["roberta mutual-information"]="0:??:0"                                ["roberta gradient"]="0:15:0" ["roberta integrated-gradient"]="0:20:0" ["roberta times-input-gradient"]="0:30:0" )

# Actual time:         ["rnn"]="0:02:0" ["roberta"]="0:02:0"
declare -A roar_time=( ["rnn"]="0:10:0" ["roberta"]="0:15:0" )

for model_type in 'rnn' 'roberta'
do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient' 'times-input-gradient'
    do
        if [ "$model_type" == "roberta" ] && [ "$importance_measure" == 'attention' ]; then
            continue
        fi

        riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)

        dependency=''

        if precompute_jobid=$(
            submit_seeds "${pre_time[$model_type $importance_measure]}" "$seeds" "importance_measure/sst_${model_type}-pre_s-%s_m-${importance_measure::1}_rs-${riemann_samples}.csv.gz" \
                --parsable \
                $(job_script gpu) \
                experiments/compute_importance_measure.py \
                --dataset sst \
                --model-type "$model_type" \
                --importance-measure "$importance_measure" \
                --importance-caching build
        ); then
            if [ ! "$precompute_jobid" == "skipping" ]; then
                echo "Submitted precompute batch job $precompute_jobid"
                dependency="--dependency=afterok:$precompute_jobid"
            fi
        else
            echo "Could not submit precompute batch job, skipping"
            break
        fi

        for k in {1..10}
        do
            if [ "$importance_measure" != 'random' ]; then
                continue
            fi

            submit_seeds ${roar_time[$model_type]} "$seeds" "roar/sst_${model_type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                $dependency \
                $(job_script gpu) \
                experiments/stanford_sentiment.py \
                --model-type "$model_type" \
                --k "$k" --recursive-step-size 1 \
                --roar-strategy count --importance-measure "$importance_measure" \
                --importance-caching use
        done

        for k in {10..100..10}
        do
            if [ "$k" -le 90 ] || [ "$importance_measure" = "random" ]; then
                submit_seeds ${roar_time[$model_type]} "$seeds" "roar/sst_${model_type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                    $dependency \
                    $(job_script gpu) \
                    experiments/stanford_sentiment.py \
                    --model-type "$model_type" \
                    --k "$k" --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure "$importance_measure" \
                    --importance-caching use
            fi
        done
    done
done
