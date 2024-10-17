#!/bin/bash

# Submit the - vary the dt

export TSTEPS="0.1 0.01 0.001 0.0001 0.00001 0.000001"

ITER=0
for tstep in ${TSTEPS}
do
    echo ${tstep}
    if (($ITER==0)); then
        jobid=$(qsub -v idx=0,p=0,dt=${tstep} ns_runner.sh)
        #jobid=$(qsub -v idx=1,p=1 ns_runner.sh)
    else
        jobid=$(qsub -W depend=afterany:${jobid} -v idx=0,p=0,dt=${tstep} ns_runner.sh)
    fi

    ITER=$(expr $ITER + 1)
done

