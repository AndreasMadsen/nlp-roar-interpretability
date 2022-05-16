#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 285
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn random"]="0:05:0"     ["rnn mutual-information"]="0:05:0"     ["rnn attention"]="0:05:0"     ["rnn gradient"]="0:05:0"     ["rnn integrated-gradient"]="0:20:0"     ["rnn times-input-gradient"]="0:??:0"
#                 ["roberta random"]="0:??:0" ["roberta mutual-information"]="0:??:0"                                ["roberta gradient"]="0:10:0" ["roberta integrated-gradient"]="2:10:0" ["roberta times-input-gradient"]="0:10:0"
declare -A time=( ["rnn random"]="0:20:0"     ["rnn mutual-information"]="0:20:0"     ["rnn attention"]="0:20:0"     ["rnn gradient"]="0:20:0"     ["rnn integrated-gradient"]="0:30:0"     ["rnn times-input-gradient"]="0:20:0"
                  ["roberta random"]="0:??:0" ["roberta mutual-information"]="0:??:0"                                ["roberta gradient"]="0:30:0" ["roberta integrated-gradient"]="3:00:0" ["roberta times-input-gradient"]="0:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        for importance_measure in 'attention' 'gradient' 'integrated-gradient' 'times-input-gradient'
        do
            if [ "$model_type" == "roberta" ] && [ "$importance_measure" == 'attention' ]; then
                continue
            fi

            riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)

            dependency=''

            for k in {1..10}
            do
                if last_jobid=$(
                        submit_seeds "${time[$model_type $importance_measure]}" "$seed" "roar/imdb_${model_type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/imdb.py --recursive \
                            --model-type "$model_type" \
                            --k "$k" --recursive-step-size 1 \
                            --roar-strategy count --importance-measure "$importance_measure"
                ); then
                    if [ ! "$last_jobid" == "skipping" ]; then
                        echo "Submitted batch job $last_jobid"
                        dependency="--dependency=afterok:$last_jobid"
                    fi
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            done

            dependency=''

            for k in {10..90..10}
            do
                if last_jobid=$(
                        submit_seeds "${time[$model_type $importance_measure]}" "$seed" "roar/imdb_${model_type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/imdb.py --recursive \
                            --model-type "$model_type" \
                            --k "$k" --recursive-step-size 10 \
                            --roar-strategy quantile --importance-measure "$importance_measure"
                ); then
                    if [ ! "$last_jobid" == "skipping" ]; then
                        echo "Submitted batch job $last_jobid"
                        dependency="--dependency=afterok:$last_jobid"
                    fi
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            done
        done
    done
done
