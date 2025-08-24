#!/usr/bin/env bash

# Run scripts/run_all.py in a loop and restart on exit,
# with a short delay to avoid a tight loop

while true; do
  python3 scripts/run_all.py 2>&1 | tee -a run_all.log
  sleep 1
done
