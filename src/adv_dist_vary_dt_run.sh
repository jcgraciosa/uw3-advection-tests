#!/bin/bash

# coarse test
#TSTEPS=("0.1" "0.01" "0.001" "0.0001")
#MAXSTEPS=("5" "50" "50" "50")
#NJOBS=("1" "1" "10" "100")
#LEN=4

# refined test
TSTEPS=("0.00125" "0.0025" "0.005" "0.00625" "0.0125" "0.025" "0.05")
MAXSTEPS=("50" "50" "50" "40" "40" "20" "10")
NJOBS=("8" "4" "2" "2" "1" "1" "1")
LEN=7

for ((i=0;i<${LEN};i++)) # C-like syntax works!
do
    tstep=${TSTEPS[$i]}
    N=${NJOBS[$i]}
    maxsteps=${MAXSTEPS[$i]}
    echo ${i} ${tstep} ${N} ${maxstep}

    if (($i==0)); then
        echo $i "start"
        jobid=$(qsub -v idx=0,p=0,dt=${tstep},ms=${maxsteps} ns_runner.sh)
        for ((j=1;j<$N;j++)) # C-like syntax works!
        do
            echo $i $j
            jobid=$(qsub -W depend=afterany:${jobid} -v idx=$j,p=1,dt=${tstep},ms=${maxsteps} ns_runner.sh)
        done
    else
        echo $i "start"
        jobid=$(qsub -W depend=afterany:${jobid} -v idx=0,p=0,dt=${tstep},ms=${maxsteps} ns_runner.sh)
        for ((j=1;j<$N;j++))
        do
            echo $i $j
            jobid=$(qsub -W depend=afterany:${jobid} -v idx=$j,p=1,dt=${tstep},ms=${maxsteps} ns_runner.sh)
        done
    fi
done

