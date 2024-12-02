#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p v6_384
#SBATCH -J LR_Si-lda222
#SBATCH --nodes=6
##SBATCH -x cpu01
##SBATCH --ntasks=16
#SBATCH --cpus-per-task=96
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Working directory is $PWD
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo Begin Time: `date`
workdir=$(basename `pwd`)
### * * * Running the tasks * * * ###

# abacus_work=/home/linpz/software/abacus-develop/bin_save/ABACUS.mpi-2022.10.19-9727f5
# mpirun -n 1 -env OMP_NUM_THREADS=64 $abacus_work  >job.log 2>job.err
 
abacus_work=/public1/home/t6s000394/jghan/software/abacus-develop/rdmft-abacus/bin/abacus
mpirun -n 6 -ppn 1 -env OMP_NUM_THREADS=96 $abacus_work  >job.log 2>job.err

echo End Time: `date`
