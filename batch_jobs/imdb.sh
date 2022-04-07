#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time: "0:04:0"
submit_seeds "0:15:0" "$seeds" "roar/imdb_rnn_s-%s.json" \
    --mem=6G \
    $(job_script gpu) \
    experiments/imdb.py

# Actual time: "0:26:0"
submit_seeds "0:35:0" "$seeds" "roar/imdb_roberta_s-%s.json" \
    --mem=64G \
    $(job_script gpu) \
    experiments/imdb.py \
    --model-type roberta --batch-size 8 --max-epochs 3
