
job_script () {
    loginnode=${HOSTNAME%%.*}
    cluster=${loginnode//[0-9]/}
    jobscript="python_${cluster}_$1_job.sh"

    if [ ! -f "$jobscript" ]; then
        echo "python_${cluster}_$1_job.sh not found"
        exit 1
    fi

    echo "$jobscript"
}
