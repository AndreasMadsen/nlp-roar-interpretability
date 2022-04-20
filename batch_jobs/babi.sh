#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn 1"]="0:07:0"     ["rnn 2"]="0:10:0"     ["rnn 3"]="0:22:0"
# Actual time:    ["roberta 1"]="0:04:0" ["roberta 2"]="0:06:0" ["roberta 3"]="0:11:0"
declare -A time=( ["rnn 1"]="0:20:0"     ["rnn 2"]="0:30:0"     ["rnn 3"]="0:40:0"
                  ["roberta 1"]="0:20:0" ["roberta 2"]="0:20:0" ["roberta 3"]="0:30:0" )

for model_type in 'rnn' 'roberta'
do
    for type in 1 2 3
    do
        submit_seeds "${time[$model_type $type]}" "$seeds" "roar/babi-${type}_${model_type}_s-%s.json" \
            $(job_script gpu) \
            experiments/babi.py \
            --task "$type" \
            --model-type "$model_type"
    done
done
