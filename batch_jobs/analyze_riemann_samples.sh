#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0"

declare -A pre_time=(
    ["2 sst"]="0:15:0"   ["2 snli"]="0:40:0"   ["2 imdb"]="0:30:0"   ["2 babi-1"]="0:15:0"   ["2 babi-2"]="0:15:0"   ["2 babi-3"]="0:20:0"   ["2 mimic-a"]="0:55:0"   ["2 mimic-d"]="1:50:0"
    ["5 sst"]="0:15:0"   ["5 snli"]="0:40:0"   ["5 imdb"]="0:30:0"   ["5 babi-1"]="0:15:0"   ["5 babi-2"]="0:15:0"   ["5 babi-3"]="0:20:0"   ["5 mimic-a"]="0:55:0"   ["5 mimic-d"]="1:50:0"
    ["10 sst"]="0:15:0"  ["10 snli"]="0:40:0"  ["10 imdb"]="0:30:0"  ["10 babi-1"]="0:15:0"  ["10 babi-2"]="0:15:0"  ["10 babi-3"]="0:20:0"  ["10 mimic-a"]="0:55:0"  ["10 mimic-d"]="1:50:0"
    ["20 sst"]="0:15:0"  ["20 snli"]="0:40:0"  ["20 imdb"]="0:30:0"  ["20 babi-1"]="0:15:0"  ["20 babi-2"]="0:15:0"  ["20 babi-3"]="0:20:0"  ["20 mimic-a"]="0:55:0"  ["20 mimic-d"]="1:50:0"
    ["30 sst"]="0:15:0"  ["30 snli"]="0:40:0"  ["30 imdb"]="0:30:0"  ["30 babi-1"]="0:15:0"  ["30 babi-2"]="0:15:0"  ["30 babi-3"]="0:20:0"  ["30 mimic-a"]="0:55:0"  ["30 mimic-d"]="1:50:0"
    ["40 sst"]="0:15:0"  ["40 snli"]="0:40:0"  ["40 imdb"]="0:30:0"  ["40 babi-1"]="0:15:0"  ["40 babi-2"]="0:15:0"  ["40 babi-3"]="0:20:0"  ["40 mimic-a"]="0:55:0"  ["40 mimic-d"]="1:50:0"
    ["50 sst"]="0:15:0"  ["50 snli"]="0:40:0"  ["50 imdb"]="0:30:0"  ["50 babi-1"]="0:15:0"  ["50 babi-2"]="0:15:0"  ["50 babi-3"]="0:20:0"  ["50 mimic-a"]="0:55:0"  ["50 mimic-d"]="1:50:0"
    ["60 sst"]="0:20:0"  ["60 snli"]="1:20:0"  ["60 imdb"]="0:30:0"  ["60 babi-1"]="0:15:0"  ["60 babi-2"]="0:15:0"  ["60 babi-3"]="0:20:0"  ["60 mimic-a"]="1:20:0"  ["60 mimic-d"]="2:10:0"
    ["70 sst"]="0:20:0"  ["70 snli"]="1:20:0"  ["70 imdb"]="0:30:0"  ["70 babi-1"]="0:15:0"  ["70 babi-2"]="0:15:0"  ["70 babi-3"]="0:20:0"  ["70 mimic-a"]="1:20:0"  ["70 mimic-d"]="2:10:0"
    ["80 sst"]="0:20:0"  ["80 snli"]="1:20:0"  ["80 imdb"]="0:30:0"  ["80 babi-1"]="0:15:0"  ["80 babi-2"]="0:15:0"  ["80 babi-3"]="0:20:0"  ["80 mimic-a"]="1:20:0"  ["80 mimic-d"]="2:10:0"
    ["90 sst"]="0:20:0"  ["90 snli"]="1:20:0"  ["90 imdb"]="0:30:0"  ["90 babi-1"]="0:15:0"  ["90 babi-2"]="0:15:0"  ["90 babi-3"]="0:20:0"  ["90 mimic-a"]="1:20:0"  ["90 mimic-d"]="2:10:0"
    ["100 sst"]="0:20:0" ["100 snli"]="1:20:0" ["100 imdb"]="0:30:0" ["100 babi-1"]="0:15:0" ["100 babi-2"]="0:15:0" ["100 babi-3"]="0:20:0" ["100 mimic-a"]="1:20:0" ["100 mimic-d"]="2:10:0"
)

declare -A pre_memory=(
    ["sst"]="6G" ["snli"]="24G" ["imdb"]="6G" ["babi-1"]="6G" ["babi-2"]="6G" ["babi-3"]="6G" ["mimic-a"]="8G" ["mimic-d"]="8G"
)

for seed in $(echo "$seeds")
do
    for dataset in 'sst' 'snli' 'imdb' 'babi-1' 'babi-2' 'babi-3' 'mimic-a' 'mimic-d'
    do
        for riemann_samples in 2 5 {10..100..10}
        do
            submit_seeds ${pre_time[$riemann_samples $dataset]} "$seed" "importance_measure/${dataset}-pre_s-%s_m-i_rs-${riemann_samples}.csv.gz" \
                --mem=${pre_memory[$dataset]} --parsable \
                $(job_script gpu) \
                experiments/compute_importance_measure.py \
                --dataset "$dataset" \
                --importance-measure integrated-gradient \
                --riemann-samples "$riemann_samples"
        done
    done
done
