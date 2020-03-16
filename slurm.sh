#!/bin/bash

#SBATCH --job-name bn0_ex0job                                        # Job name

### Logging
#SBATCH --output=logs/grid_train_job_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=logs/grid_train_job_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=tesca.fitzgerald@gmail.com  # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE                                      

### Node info
###SBATCH --partition Test                                             # Queue name [NOT NEEDED FOR NOW]
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                        # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 48:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:1                                                       # Number of gpus needed
#SBATCH --mem=6G                                                         # Memory requirements
#SBATCH --cpus-per-task=8                                              # Number of cpus needed per task

python -u grid_train.py --epoch=50001 --task_num=10 --bn=$1 --exclude=$2 | tee /u/tesca/data/logs/grid_bn$1_ex$2-r2.txt
