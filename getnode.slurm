#!/bin/bash
#SBATCH --job-name=capstone_cv
#SBATCH --open-mode=append
#SBATCH --output=/scratch/alc9635/cds-bootcamp/homework/cv/logs/capstone_cv.out
#SBATCH --error=/scratch/alc9635/cds-bootcamp/homework/cv/logs/capstone_cv.err
#SBATCH --export=ALL
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH -c 8
sleep 5h
