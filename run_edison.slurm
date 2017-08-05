#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --job-name=mpishm

# THIS MUST BE THE SAME AS --nodes ABOVE
NODES=2

# processes per node
NODE_PROC=12
NPROC=$(( NODES * NODE_PROC ))

# logical cores to allocate per MPI rank
NODE_PHYS=24
NODE_DEPTH=$(( NODE_PHYS / NODE_PROC ))

# only use one thread per physical core.
NODE_CORE=24
NODE_THREAD=$(( NODE_CORE / NODE_PROC ))

export OMP_NUM_THREADS=${NODE_THREAD}

# treat each hyperthread as a place an OpenMP task can go
export OMP_PLACES=threads

# spread threads as widely as possible (avoid sharing cores,cache etc)
export OMP_PROC_BIND=spread

com="srun -n ${NPROC} -N ${NODES} -c ${NODE_DEPTH} --cpu_bind=cores \
python3 test.py"

echo ${com}
eval ${com}

