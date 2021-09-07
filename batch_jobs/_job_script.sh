
job_script () {
    local loginnode=${HOSTNAME%%.*}
    local cluster=${loginnode//[0-9]/}
    local jobscript="python_${cluster}_$1_job.sh"

    if [ ! -f "$jobscript" ]; then
        echo "python_${cluster}_$1_job.sh not found" 1>&2
        return 1
    fi

    echo "$jobscript"
}

function join_by {
    local IFS="$1";
    shift;
    echo "$*";
}

submit_seeds () {
    local walltime=$1
    local seeds=$2
    local name=$3

    local run_seeds=()
    local filename
    for seed in $(echo "$seeds")
    do
        filename=$(printf $SCRATCH"/nlproar/results/$name" "$seed")
        if [ ! -f "$filename" ]; then
            run_seeds+=($seed)
            echo "scheduling $filename" 1>&2
        fi
    done

    if [ ! "${#run_seeds[@]}" -eq 0 ]; then
        local walltime_times_nb_seeds=$(python3 -c \
        "from datetime import datetime; \
         t = (datetime.strptime('$walltime', '%H:%M:%S') - datetime.strptime('0:0:0', '%H:%M:%S')) * ${#run_seeds[@]}; \
         print(':'.join(map(str, [*divmod(int(t.total_seconds()) // 60, 60), 0])));
        ")

        local concat_seeds=$(join_by '' "${run_seeds[@]}")
        local jobname=$(basename "$name")
        jobname=${jobname%.*}
        jobname=${jobname%.*}
        jobname=$(printf "$jobname" "$concat_seeds")
        sbatch --time="$walltime_times_nb_seeds" \
               --export=ALL,RUN_SEEDS="$(join_by ' ' "${run_seeds[@]}")" \
               -J "$jobname" \
               -o "$SCRATCH"/nlproar/logs/%x.%j.out -e "$SCRATCH"/nlproar/logs/%x.%j.err \
               "${@:4}"
    else
        echo "skipping"
    fi
}
