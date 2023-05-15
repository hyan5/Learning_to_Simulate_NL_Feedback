#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=text_gen
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --output=text_gen.%j.out
#SBATCH --error=text_gen.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL 
#SBATCH --time=2-00:00:00

## Slurm can send you updates via email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hyan5@gmu.edu

## Load the relevant modules needed for the job
module load gnu10/10.3.0
source ispenv/bin/activate
## Run your program or script
cd /scratch/hyan5/interaction_semantic_parsing_with_NL_feedback/feedback_simulation/Correction_Model
export CORR_HOME=$(pwd)
export PYTHONPATH=$CORR_HOME:$CORR_HOME/evaluation:$PYTHONPATH
python train.py --data_dir=data/splash --data_revision=fweqs --model=t5-base
# python train.py --data_dir=data/splash --data_revision=tqes --model=facebook/bart-large