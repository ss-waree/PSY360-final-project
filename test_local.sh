#!/bin/bash

# Array of alpha values
alphas=(0.5 2.0 10.0)

# Array of beta combinations
betas=(
    "0.2 0.8"
    "0.5 0.5"
    "0.8 0.2"
)

# Array of num_black and num_white combinations
num_combinations=(
    "1 0"
    "0 1"
    "2 0"
    "0 2"
    "10 0"
    "0 10"
    "20 0"
    "0 20"
    "1 1"
    "2 2"
    "5 5"
    "10 10"
    "2 1"
    "1 2"
    "4 2"
    "2 4"
    "8 4"
    "4 8"
    "10 11"
    "11 10"
    "1 4"
    "4 1"
    "2 8"
    "8 2"
    "4 16"
    "16 4"
)

# DON'T FORGET TO CHANGE THE LETTER IN FRONT OF MODEL NAME HERE AND IN EVAL.PY

# Activate the virtual environment
source .venv/bin/activate

# Loop over alpha values
for alpha in "${alphas[@]}"; do
    # Loop over beta combinations
    for beta in "${betas[@]}"; do
        # Split beta into beta0 and beta1
        read beta0 beta1 <<<"$beta"

        # Loop over num_black and num_white combinations
        for num in "${num_combinations[@]}"; do
            # Split num into num_black and num_white
            read num_black num_white <<<"$num"

            # Construct model name
            model_name="a_${alpha}_${beta0}_${beta1}_0"

            # Run the python script with the constructed model name and num_black/num_white
            python meta_train.py --min_n_features 1 --max_n_features 1 --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset marble --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name "${model_name}" --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --eval --eval_marble --marble_n_runs 100 --num_black "${num_black}" --num_white "${num_white}"
        done
    done
done

# Deactivate the virtual environment after the job is done
deactivate
