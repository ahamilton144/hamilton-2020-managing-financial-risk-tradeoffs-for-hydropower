#!/bin/bash
#SBATCH --job-name="sensitivity"
#SBATCH --output="sensitivity_%j.out"
#SBATCH --partition=normal
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH -t 10:00:00

NFE=10000
seed_sample=1

time {
  for seed_borg in {1..10}
  do
    echo
    echo "Seed "${seed_borg}
    time mpirun main_sensitivity_borgms $seed_borg $seed_sample $NFE 
  done
  echo
  echo "Finished all seeds "
}

