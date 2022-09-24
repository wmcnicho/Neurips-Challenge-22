#!/bin/bash
#SBATCH --gres gpu:5
#SBATCH --cpus-per-task=3
#SBATCH --mem=100000
#SBATCH --mail-type=END
#SBATCH --mail-user=jaewooklee@umass.edu
#SBATCH --partition=gypsum-rtx8000
#SBATCH --constraint=""
#SBATCH --time=7-00
#SBATCH -o /work/jaewooklee_umass_edu/neurips_2022/sanity_check/output/%j.out
#SBATCH -e /work/jaewooklee_umass_edu/neurips_2022/sanity_check/error/%j.out

while getopts b:c:q:s:t:u:l:e: option
do
    case "${option}"
        in        
	    b)batch=${OPTARG};;
        c)num_constructs=${OPTARG};;
        q)num_questions=${OPTARG};;
        s)num_students=${OPTARG};;
        t)temperature=${OPTARG};;
        u)unroll=${OPTARG};;
        l)learning_rate=${OPTARG};;
        e)num_epochs=${OPTARG};;
    esac
done

python3 /work/jaewooklee_umass_edu/neurips_2022/sanity_check/train.py -B $batch -C $num_constructs -Q $num_questions -S $num_students -T $temperature -U $unroll -L $learning_rate -E $num_epochs
