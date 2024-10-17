#!/bin/bash

# Submit the first job
jobid=$(qsub -v idx=0,p=0 ns_runner.sh)
#jobid=$(qsub -v idx=1,p=1 ns_runner.sh)

# Submit the remaining jobs in a loop
#for i in {1..4}
#do
#  jobid=$(qsub -W depend=afterany:${jobid} -v idx=$i,p=1 ns_runner.sh)
#done

