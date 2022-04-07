#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time: "0:01:30"
submit_seeds "0:10:0" "$seeds" "roar/sst_rnn_s-%s.json" \
    --mem=6G \
    $(job_script gpu) \
    experiments/stanford_sentiment.py

# Actual time: "0:01:30"
submit_seeds "0:30:0" "$seeds" "roar/sst_roberta_s-%s.json" \
    --mem=64G \
    $(job_script gpu) \
    experiments/stanford_sentiment.py \
    --model-type roberta --batch-size 8 --max-epochs 3
