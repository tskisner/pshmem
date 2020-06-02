#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --constraint=knl,quad,cache
#SBATCH --nodes=2
#SBATCH --core-spec=4
#SBATCH --time=00:10:00
#SBATCH --job-name=pshmem

# Nodes used in the job
NODES=${SLURM_JOB_NUM_NODES}

# set procs and threads
NODE_SLOTS=256
NODE_PROC=64
PROC_THREADS=1
PROC_DEPTH=$(( NODE_SLOTS / NODE_PROC ))

# total number of processes on all nodes
NPROC=$(( NODES * NODE_PROC ))

echo "Using ${NODES} node(s), which have ${NODE_SLOTS} thread slots each."
echo "Starting ${NODE_PROC} process(es) per node (${NPROC} total), each with ${PROC_THREADS} OpenMP threads."

export OMP_NUM_THREADS=${PROC_THREADS}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

com="srun -n ${NPROC} -N ${NODES} -c ${PROC_DEPTH} --cpu_bind=cores \
python3 -c 'import pshmem; pshmem.test()'"

echo ${com}
eval ${com}
