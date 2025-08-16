#!/bin/bash

# Default values for N and X
N=10
X=100

# Parse command-line arguments for N and X
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_iterations) N="$2"; shift ;;
        --num_games) X="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Step 0: Delete specific files if they exist
rm -rf out/*
rm -f training_data.bin alpha_zero.log model_eval.log training.log

# Step 1: Create a starting model
python scripts/model.py

# Step 2: Run baseline evaluation
./build/model_eval

# Step 3: Loop N times
for ((i=1; i<=N; i++)); do
    # Step 3a: Run alpha_zero with X games
    ./build/alpha_zero --num_games="$X"

    # Step 3b: Train the model
    python scripts/train.py

    # Step 3c: Copy the latest model to model.pt
    latest_model=$(ls out/model_*.pt | sort -V | tail -n 1)
    cp "$latest_model" ./model.pt

    # Step 3d: Run model evaluation
    ./build/model_eval
done
