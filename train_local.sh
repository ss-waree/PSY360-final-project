#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Array of alpha values
alphas=(0.5 2.0 10.0)

# Array of beta combinations
betas=(
    "0.2 0.8"
    "0.5 0.5"
    "0.8 0.2"
)

# Loop over alpha values
for alpha in "${alphas[@]}"; do
    # Loop over beta combinations
    for beta in "${betas[@]}"; do
        # Split beta into beta0 and beta1
        read beta0 beta1 <<<"$beta"

        # Construct model name
        model_name="b_${alpha}_${beta0}_${beta1}"

        python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name "${model_name}" --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --alpha "${alpha}" --beta_0 "${beta0}" --beta_1 "${beta1}" 

    done
done

# Deactivate the virtual environment after the job is done
deactivate
