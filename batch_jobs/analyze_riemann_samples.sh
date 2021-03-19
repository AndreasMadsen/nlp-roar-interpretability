#!/bin/bash
# Actual time:    ["2"]="0:21:0"   ["5"]="0:31:0" ["10"]="0:45:0" ["20"]="1:23:0"
#                 ["30"]="1:54:0" ["40"]="2:26:0" ["50"]="3:13:0" ["60"]="3:35:0"
#                 ["70"]="4:06:0" ["80"]="4:38:0" ["90"]="5:13:0" ["100"]="5:46:0"
declare -A time=( ["2"]="0:40:0"   ["5"]="0:50:0" ["10"]="1:00:0" ["20"]="1:50:0"
                  ["30"]="2:20:0" ["40"]="2:50:0" ["50"]="3:40:0" ["60"]="4:00:0"
                  ["70"]="4:30:0" ["80"]="5:00:0" ["90"]="5:40:0" ["100"]="6:10:0" )

for k in 2 5 {10..100..10}
do
    sbatch --time=${time[$k]} --mem=16G \
        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
        -J "importance_s-0_m-i_rs-${k}" ./python_job.sh \
        experiments/analyze_importance_measure.py \
        --seed 0 \
        --importance-measure integrated-gradient \
        --riemann-samples ${k}
done
