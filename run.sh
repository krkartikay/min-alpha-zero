#!/usr/bin/env bash

# Run scripts/run_all.py in a loop and restart on exit,
# with a short delay to avoid a tight loop

while true; do
  python3 scripts/run_all.py 2>&1 | tee -a run_all.log
  dataset_size=$(python3 -c "from scripts.dataset import TrainingDataset; print(len(TrainingDataset('training_data.bin')))")
  echo -e "\n\nCurrent TrainingDataset size: $dataset_size\n\n" | tee -a run_all.log
  sleep 1
done
