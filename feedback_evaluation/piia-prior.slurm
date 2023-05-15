#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=text_eval
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --output=text_eval.%j.out
#SBATCH --error=text_eval.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL 
#SBATCH --time=3-00:00:00

## Slurm can send you updates via email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hyan5@gmu.edu

## Load the relevant modules needed for the job
module load gnu10/10.3.0
source ispenv/bin/activate

## Run your program or script
cd /scratch/hyan5/interaction_semantic_parsing_with_NL_feedback/feedback_evaluation
python train_spider_aligner.py