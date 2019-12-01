#!/bin/bash
#SBATCH --account=research
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -J test
#SBATCH -p 236605

# And finally run the job?
#python -u pre_processing.py
#python -u pre_process_2.py
#python -u train_model.py
python -u test_model.py
python -u stiching.py
python -u evaluate.py