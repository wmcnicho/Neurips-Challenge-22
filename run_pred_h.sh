#!/bin/bash
#SBATCH --gres gpu:5
#SBATCH --cpus-per-task=3
#SBATCH --mem=100000
#SBATCH --mail-type=END
#SBATCH --mail-user=wmcnichols@umass.edu
#SBATCH --partition=gypsum-rtx8000
#SBATCH --constraint=""
#SBATCH --time=7-00
#SBATCH -o /home/wmcnichols_umass_edu/Neurips-Challenge-22/output/%j.out
#SBATCH -e /home/wmcnichols_umass_edu/Neurips-Challenge-22/error/%j.out

python3 /home/wmcnichols_umass_edu/Neurips-Challenge-22/predict_graph.py
