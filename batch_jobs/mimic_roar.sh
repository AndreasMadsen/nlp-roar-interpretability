#!/bin/bash
# jobs: 5 * 2 * 4 * (10 + 9) = 760
source "batch_jobs/_job_script.sh"

declare -A time=( ["anemia random"]="0:25:0"   ["anemia attention"]="0:25:0"   ["anemia gradient"]="0:30:0" ["anemia integrated-gradient"]="0:45:0"
                  ["diabetes random"]="0:40:0" ["diabetes attention"]="0:40:0" ["diabetes gradient"]="0:50:0" ["diabetes integrated-gradient"]="1:10:0" )

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
        do
            if precompute_jobid=$(
                sbatch --time=2:00:0 --mem=8G --parsable \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J "importance_s-${seed}_m-${importance_measure::1}" ./python_job.sh \
                    experiments/compute_importance_measure.py \
                    --seed ${seed} \
                    --dataset "mimic-${subset::1}" \
                    --importance-measure ${importance_measure} \
                    --importance-caching build
            );  then
                echo "Submitted precompute batch job ${precompute_jobid}"
            else
                echo "Could not submit precompute batch job, skipping"
                break
            fi

            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$subset $importance_measure]} --mem=8G --dependency=afterok:${precompute_jobid} \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0 $(job_script gpu) \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 1 \
                        --roar-strategy count --importance-measure ${importance_measure} \
                        --importance-caching use \
                        --subset ${subset}
                fi
            done

            for k in {10..90..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$subset $importance_measure]} --mem=8G --dependency=afterok:${precompute_jobid} \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 $(job_script gpu) \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 10 \
                        --roar-strategy quantile --importance-measure ${importance_measure} \
                        --importance-caching use \
                        --subset ${subset}
                fi
            done
        done
    done
done
