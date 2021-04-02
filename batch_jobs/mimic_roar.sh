#!/bin/bash
# jobs: 5 * 2 * 4 * (10 + 9) = 760
source "batch_jobs/_job_script.sh"

# Actual   pre_time=( ["anemia random"]="0:04:0"   ["anemia attention"]="0:05:0"   ["anemia gradient"]="0:07:0"   ["anemia integrated-gradient"]="0:36:0"
#                     ["diabetes random"]="0:05:0" ["diabetes attention"]="0:06:0" ["diabetes gradient"]="0:12:0" ["diabetes integrated-gradient"]="1:29:0" )
declare -A pre_time=( ["anemia random"]="0:15:0"   ["anemia attention"]="0:15:0"   ["anemia gradient"]="0:20:0"   ["anemia integrated-gradient"]="0:55:0"
                      ["diabetes random"]="0:20:0" ["diabetes attention"]="0:20:0" ["diabetes gradient"]="0:30:0" ["diabetes integrated-gradient"]="1:50:0" )

# Actual   roar_time=( ["anemia"]="0:09:0"   ["diabetes"]="0:17:0" )
declare -A roar_time=( ["anemia"]="0:20:0"   ["diabetes"]="0:30:0" )


for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
        do
            riemann_samples=$(( $importance_measure == integrated-gradient ? 50 : 0 ))

            if precompute_jobid=$(
                sbatch --time=${pre_time[$subset $importance_measure]} --mem=8G --parsable \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J "mimic-${subset::1}-pre_s-${seed}_m-${importance_measure::1}_r-0_rs-${riemann_samples}" $(job_script gpu) \
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
<<<<<<< HEAD
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-50.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-50
=======
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}
>>>>>>> 53430b6... set rs-0 for non integrated-gradient measures
                    sbatch --time=${roar_time[$subset]} --mem=8G --dependency=afterok:${precompute_jobid} \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples} $(job_script gpu) \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 1 \
                        --roar-strategy count --importance-measure ${importance_measure} \
                        --importance-caching use \
                        --subset ${subset}
                fi
            done

            for k in {10..90..10}
            do
<<<<<<< HEAD
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-50.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-50
=======
                if [ ! -f $SCRATCH"/comp550/results/roar/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}
>>>>>>> 53430b6... set rs-0 for non integrated-gradient measures
                    sbatch --time=${roar_time[$subset]} --mem=8G --dependency=afterok:${precompute_jobid} \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples} $(job_script gpu) \
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
