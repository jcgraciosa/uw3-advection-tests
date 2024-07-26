#!/bin/bash

##PBS -P m18
#PBS -P el06
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l mem=192GB
#PBS -l jobfs=1GB
#PBS -l ncpus=8
#PBS -l software=underworld
#PBS -l wd
#PBS -M juancarlos.graciosa@anu.edu.au

#PBS -l storage=gdata/m18+scratch/el06
##PBS -l storage=gdata/m18

#export MODULEPATH=/g/data/m18/modulefiles:$MODULEPATH
#module load openmpi/4.1.4 petsc/3.18.1 intel-mkl/2021.4.0 python3/3.10.4 python3-as-python gmsh/4.4.1
#module load underworld/3.0.0

# update pythonpath
#cd /home/157/jg0883/underworld3
#source pypathsetup.sh
#cd -

# disable numpy intrinsic parallelism
#export OPENBLAS_NUM_THREADS=1

# disable file locking in hdf5
#export HDF5_USE_FILE_LOCKING=FALSE

# warning
#export H5PY_DEFAULT_READONLY=1
source params.sh
source /home/157/jg0883/install-scripts/gadi_install.sh
env
cat Test_AdvVec_run2.py

# execution
#mpiexec -x LD_PRELOAD=libmpi.so python3 test_read.py
#mpiexec -x LD_PRELOAD=libmpi.so python3 Test_Advect_VectTensor_run.py --idx $idx --prev $p # without any dt
#mpiexec -x LD_PRELOAD=libmpi.so python3 Test_Advect_VectTensor_run.py --idx $idx --prev $p --dt $dt
mpiexec -x LD_PRELOAD=libmpi.so python3 Test_AdvVec_run2.py --idx $idx --prev $p --dt $dt --ms $ms > ${MESH_USE}_dt${dt}_${idx}.log

## Dan's
##PBS -P ty11
##PBS -l jobfs=100MB
##PBS -l ncpus=24
##PBS -l wd
##PBS -l storage=gdata/m18+scratch/ty11
