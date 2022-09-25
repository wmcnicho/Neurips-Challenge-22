#!/bin/bash
#SBATCH -c 16
#SBATCH --cpus-per-task=3
#SBATCH --mem=100000
#SBATCH --mail-type=END
#SBATCH --mail-user=jaewooklee@umass.edu
#SBATCH --constraint=""
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o /work/jaewooklee_umass_edu/neurips_2022/sanity_check/output/%j.out
#SBATCH -e /work/jaewooklee_umass_edu/neurips_2022/sanity_check/error/%j.out

#python3 /work/jaewooklee_umass_edu/neurips_2022/sanity_check/train.py  -B 4 -C 4 -Q 500 -S 500 -T 20 -U 100 -L 0.001 -E 200

python3 /work/jaewooklee_umass_edu/neurips_2022/sanity_check/train.py  -B 4 -C 4 -Q 500 -S 500 -T 20 -U 100 -L 0.001 -E 200 -P
