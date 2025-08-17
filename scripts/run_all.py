#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import min_alpha_zero as maz
from train import main as train_main

NUM_SIMULATIONS = 1000
NUM_GAMES_EVAL = 10
NUM_GAMES_SELFPLAY = 100


def setup_config():
    config = maz.get_config()
    config.channel_size = 128
    config.num_simulations = NUM_SIMULATIONS
    config.batch_size = 1000
    config.num_games = NUM_GAMES_SELFPLAY
    config.model_path = "model.pt"
    config.training_file = "training_data.bin"
    config.debug = False
    config.temperature = 1.0
    return config


def run_model_eval():
    print("\n" + "=" * 50)
    print("Running Model Evaluation")
    print("=" * 50)
    config = maz.get_config()
    config.num_games = NUM_GAMES_EVAL
    agent1 = maz.RawModelAgent()
    agent2 = maz.RandomAgent()
    maz.run_agent_tournament(agent1, agent2)


def run_self_play():
    print("\n" + "=" * 50)
    print("Running Self-play (100 games)")
    print("=" * 50)
    config = maz.get_config()
    config.num_games = NUM_GAMES_SELFPLAY
    maz.run_worker()


def run_training():
    print("\n" + "=" * 50)
    print("Running Model Training")
    print("=" * 50)
    train_main()


def main():
    iteration = 1

    setup_config()
    maz.init_globals()
    maz.init_model()
    maz.start_evaluator_thread()

    while True:
        print(f"\n{'#'*60}")
        print(f"ITERATION {iteration}")
        print(f"{'#'*60}")

        run_model_eval()
        run_self_play()
        run_training()

        iteration += 1


if __name__ == "__main__":
    main()
