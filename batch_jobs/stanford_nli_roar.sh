#!/bin/bash
declare -A time=( ["random"]="1:30:0" ["attention"]="1:30:0" ["gradient"]="1:40:0")

#!/bin/bash
for seed in {0..4}
do
    for importance_measure in 'random' 'attention' 'gradient'
    do
        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0.json" ]; then
                echo snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0
                sbatch --time=${time[$importance_measure]} --mem=32G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0 ./python_job.sh \
                    experiments/stanford_nli.py \
                    --seed ${seed} --k ${k} --recursive-step-size 1 \
                    --roar-strategy count --importance-measure ${importance_measure}
            fi
        done

        for k in {5..95..5}
        do
            if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                echo snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                sbatch --time=${time[$importance_measure]} --mem=32G \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 ./python_job.sh \
                    experiments/stanford_nli.py \
                    --seed ${seed} --k ${k} --recursive-step-size 5 \
                    --roar-strategy quantile --importance-measure ${importance_measure}
            fi
        done
    done
done
