#!/bin/bash

# combined

#TSTEPS=("0.5" "0.1" "0.05")
#MAXSTEPS=("1" "5" "10")
#LEN=3

TSTEPS=("0.5")
MAXSTEPS=("1")
LEN=1

source params.sh

for ((i=0;i<${LEN};i++)) # C-like syntax works!
do
    tstep=${TSTEPS[$i]}
    maxsteps=${MAXSTEPS[$i]}
    echo ${i} ${tstep} ${maxstep}

    idx=0
    p=0

    echo ${idx}
    echo ${p}

    mpirun -np 7 python3 Test_AdvVec_run2.py --idx ${idx} --prev $p --dt ${tstep} --ms ${maxsteps} > ${MESH_USE}_res${RES}_dt${tstep}_${idx}.log
    #python3 Test_AdvVec_run2.py --idx ${idx} --prev $p --dt ${tstep} --ms ${maxsteps} > ${MESH_USE}_res${RES}_dt${tstep}_${idx}.log

done

