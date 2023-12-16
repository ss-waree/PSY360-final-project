#!/bin/bash

#SBATCH --job-name=meta_train_job
#SBATCH --output=meta_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=sethapun@princeton.edu

# Activate the virtual environment
module purge
source .venv/bin/activate

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_0.5_0.2_0.8 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 0.5 --beta_0 0.2 --beta_1 0.8 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_0.5_0.5_0.5 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 0.5 --beta_0 0.5 --beta_1 0.5 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_0.5_0.8_0.2 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 0.5 --beta_0 0.8 --beta_1 0.2 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_2.0_0.2_0.8 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 2.0 --beta_0 0.2 --beta_1 0.8 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_2.0_0.5_0.5 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 2.0 --beta_0 0.5 --beta_1 0.5 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_2.0_0.8_0.2 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 2.0 --beta_0 0.8 --beta_1 0.2 &

python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name a_10.0_0.8_0.2 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha 10.0 --beta_0 0.8 --beta_1 0.2 &

wait

# Deactivate the virtual environment after the job is done
deactivate
