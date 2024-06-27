#!/bin/sh

#SBATCH --job-name=10cv # the name of the job meaning '10-fold cross validation'
#SBATCH --array=0-9 # the 'k's that is passed to the jobs and it is inclusive, e.g. 0-9 means we run 10 Jobs
#SBATCH -p gpu # GPU queue
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task 32 # CPUs per Task, if you request one GPU, it is reasonable to get 32 threads
#SBATCH --mem 120G # Memory per CPU
#SBATCH --mail-type END # This sends you an email once your job finishes
#SBATCH -o .../logs-%A-%a.out # Logs are saved in the folder '...'

echo Running Task ID $SLURM_ARRAY_TASK_ID

... -u train_model.py $SLURM_ARRAY_TASK_ID # '...' is the conda environment with which you run the script 'train_model.py'

exit