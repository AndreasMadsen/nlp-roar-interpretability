#!/bin/bash
# jobs: 5 * 3 * 3 * (10 + 9) = 855
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn 1 random"]="0:08:0"     ["rnn 1 mutual-information"]="0:07:0"     ["rnn 1 attention"]="0:09:0"     ["rnn 1 gradient"]="0:08:0"     ["rnn 1 integrated-gradient"]="0:10:0"
#                 ["rnn 2 random"]="0:11:0"     ["rnn 2 mutual-information"]="0:11:0"     ["rnn 2 attention"]="0:12:0"     ["rnn 2 gradient"]="0:12:0"     ["rnn 2 integrated-gradient"]="0:15:0"
#                 ["rnn 3 random"]="0:25:0"     ["rnn 3 mutual-information"]="0:21:0"     ["rnn 3 attention"]="0:25:0"     ["rnn 3 gradient"]="0:25:0"     ["rnn 3 integrated-gradient"]="0:32:0"
#                 ["roberta 1 random"]="0:??:0" ["roberta 1 mutual-information"]="0:??:0"                                  ["roberta 1 gradient"]="0:04:0" ["roberta 1 integrated-gradient"]="0:11:0" ["roberta 1 times-input-gradient"]="0:04:0"
#                 ["roberta 2 random"]="0:??:0" ["roberta 2 mutual-information"]="0:??:0"                                  ["roberta 2 gradient"]="0:06:0" ["roberta 2 integrated-gradient"]="0:32:0" ["roberta 2 times-input-gradient"]="0:06:0"
#                 ["roberta 3 random"]="0:??:0" ["roberta 3 mutual-information"]="0:??:0"                                  ["roberta 3 gradient"]="0:13:0" ["roberta 3 integrated-gradient"]="1:12:0" ["roberta 3 times-input-gradient"]="0:13:0"
declare -A time=( ["rnn 1 random"]="0:20:0"     ["rnn 1 mutual-information"]="0:20:0"     ["rnn 1 attention"]="0:20:0"     ["rnn 1 gradient"]="0:20:0"     ["rnn 1 integrated-gradient"]="1:20:0"     ["rnn 1 times-input-gradient"]="0:20:0"
                  ["rnn 2 random"]="0:25:0"     ["rnn 2 mutual-information"]="0:25:0"     ["rnn 2 attention"]="0:25:0"     ["rnn 2 gradient"]="0:25:0"     ["rnn 2 integrated-gradient"]="1:25:0"     ["rnn 2 times-input-gradient"]="0:25:0"
                  ["rnn 3 random"]="0:35:0"     ["rnn 3 mutual-information"]="0:35:0"     ["rnn 3 attention"]="0:35:0"     ["rnn 3 gradient"]="0:35:0"     ["rnn 3 integrated-gradient"]="1:45:0"     ["rnn 3 times-input-gradient"]="0:50:0"
                  ["roberta 1 random"]="0:??:0" ["roberta 1 mutual-information"]="0:??:0"                                  ["roberta 1 gradient"]="0:20:0" ["roberta 1 integrated-gradient"]="0:20:0" ["roberta 1 times-input-gradient"]="0:30:0"
                  ["roberta 2 random"]="0:??:0" ["roberta 2 mutual-information"]="0:??:0"                                  ["roberta 2 gradient"]="0:25:0" ["roberta 2 integrated-gradient"]="1:25:0" ["roberta 2 times-input-gradient"]="0:30:0"
                  ["roberta 3 random"]="0:??:0" ["roberta 3 mutual-information"]="0:??:0"                                  ["roberta 3 gradient"]="0:35:0" ["roberta 3 integrated-gradient"]="1:45:0" ["roberta 3 times-input-gradient"]="0:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        for type in 1 2 3
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
                        submit_seeds "${time[$model_type $type $importance_measure]}" "$seed" "roar/babi-${type}_${model_type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/babi.py --recursive \
                            --model-type "$model_type" \
                            --k "$k" --recursive-step-size 1 \
                            --roar-strategy count --importance-measure "$importance_measure" \
                            --task "$type"
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
                        submit_seeds "${time[$model_type $type $importance_measure]}" "$seed" "roar/babi-${type}_${model_type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/babi.py --recursive \
                            --model-type "$model_type" \
                            --k "$k" --recursive-step-size 10 \
                            --roar-strategy quantile --importance-measure "$importance_measure" \
                            --task "$type"
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
done
