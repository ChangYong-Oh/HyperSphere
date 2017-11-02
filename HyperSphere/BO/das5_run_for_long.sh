#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=30
#SBATCH --workdir=../../../../Experiments/Hypersphere/
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=changyong.oh0224@gmail.com
#SBATCH --output=sbatch-run-%N-%j-%a.out

srun python $1 levy $2 $3 &
srun python $1 rosenbrock $2 $3 &
srun python $1 styblinskitang $2 $3 &
srun python $1 schwefel $2 $3 &
srun python $1 perm $2 $3 &
srun python $1 michalewicz $2 $3
