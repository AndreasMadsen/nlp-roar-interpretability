#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn"]="0:04:0" ["roberta"]="0:8:0" )
declare -A time=( ["rnn"]="0:15:0" ["roberta"]="0:20:0" )

for model_type in 'rnn' 'roberta'
do
    submit_seeds ""${time[$model_type]}"" "$seeds" "roar/imdb_${model_type}_s-%s.json" \
        $(job_script gpu) \
        experiments/imdb.py \
        --model-type "$model_type"
done
