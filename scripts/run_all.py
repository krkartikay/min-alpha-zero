#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import min_alpha_zero as maz
from train import main as train_main

from analyze_logits import plot_logits_comparison as logits_main
from test.mate_in_one import test_mate_in_one as visualize_main

NUM_SIMULATIONS = 100
NUM_GAMES_EVAL = 10
NUM_GAMES_SELFPLAY = 10
NUM_THREADS = 10


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
    config.num_threads = NUM_THREADS
    agent1 = maz.MCTSAgent()
    agent2 = maz.RandomAgent()
    maz.run_agent_tournament(agent1, agent2)


def run_self_play():
    print("\n" + "=" * 50)
    print("Running Self-play (100 games)")
    print("=" * 50)
    config = maz.get_config()
    config.num_games = NUM_GAMES_SELFPLAY
    config.num_threads = NUM_THREADS
    maz.run_worker_threads()


def run_training():
    print("\n" + "=" * 50)
    print("Running Model Training")
    print("=" * 50)
    train_main()


def main():
    setup_config()
    maz.init_globals()
    maz.start_evaluator_thread()

    print(f"\n{'#'*60}")
    print(f"STARTING FULL ITERATION")
    print(f"{'#'*60}")

    maz.init_model()
    run_model_eval()
    run_self_play()
    run_training()

    visualize_main()
    logits_main()


if __name__ == "__main__":
    main()
