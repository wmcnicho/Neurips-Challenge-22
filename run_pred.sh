#!/bin/bash
#SBATCH --gres gpu:5
#SBATCH --cpus-per-task=3
#SBATCH --mem=100000
#SBATCH --mail-type=END
#SBATCH --mail-user=nashokkumar@umass.edu
#SBATCH --partition=gypsum-rtx8000
#SBATCH --constraint=""
#SBATCH --time=7-00
#SBATCH -o /work/nashokkumar_umass_edu/nipschal/Neurips-Challenge-22/output/%j.out
#SBATCH -e /work/nashokkumar_umass_edu/nipschal/Neurips-Challenge-22/error/%j.out

python /work/nashokkumar_umass_edu/nipschal/Neurips-Challenge-22/predict_graph.py