#!/bin/bash
#SBATCH --job-name="baseline"
#SBATCH --output="baseline_%j.out"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 2:00:00

NFE=10000
seed_sample=1

time {
  for seed_borg in {1..50}
  do
    echo
    echo "Seed "${seed_borg}
    time mpirun main_borgms $seed_borg $seed_sample $NFE 
  done
  echo
  echo "Finished all seeds "
}

