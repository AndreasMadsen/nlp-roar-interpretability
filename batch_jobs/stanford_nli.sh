#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn"]="0:44:0" ["roberta"]="2:03:0" )
declare -A time=( ["rnn"]="1:10:0" ["roberta"]="2:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        submit_seeds "${time[$model_type]}" "$seed" "roar/snli_${model_type}_s-%s.json" \
            $(job_script gpu) \
            experiments/stanford_nli.py \
            --model-type "$model_type"
    done
done
