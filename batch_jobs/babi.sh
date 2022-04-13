#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:        ["1"]="0:07:0" ["2"]="0:10:0" ["3"]="0:22:0"
declare -A time_rnn=( ["1"]="0:20:0" ["2"]="0:30:0" ["3"]="0:40:0")

# Actual time:        ["1"]="0:09:0" ["2"]="0:33:0" ["3"]="0:32:0"
declare -A time_rob=( ["1"]="0:20:0" ["2"]="0:40:0" ["3"]="0:40:0")

for type in 1 2 3
do
    submit_seeds "${time_rnn[$type]}" "$seeds" "roar/babi-${type}_rnn_s-%s.json" \
        --mem=6G \
        $(job_script gpu) \
        experiments/babi.py \
        --task "$type"

    submit_seeds "${time_rob[$type]}" "$seeds" "roar/babi-${type}_roberta_s-%s.json" \
        --mem=64G \
        $(job_script gpu) \
        experiments/babi.py \
        --task "$type" \
        --model-type roberta --batch-size 8 --max-epochs 3
done
