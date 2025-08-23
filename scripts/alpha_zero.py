#!/usr/bin/env python3
import argparse
import min_alpha_zero as maz


def main():
    parser = argparse.ArgumentParser(description="AlphaZero self-play")
    parser.add_argument(
        "--channel_size",
        type=int,
        default=128,
        help="Channel size (must be power of 2)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=200,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of nodes to process at once",
    )
    parser.add_argument(
        "--eval_timeout_ms",
        type=int,
        default=1,
        help="Timeout for evaluation requests in milliseconds",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="Number of games to play in self-play per thread",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for self-play",
    )
    parser.add_argument(
        "--model_path", type=str, default="model.pt", help="Path to the model file"
    )
    parser.add_argument(
        "--training_file",
        type=str,
        default="training_data.bin",
        help="File to store training data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to dump game tree to file after each move",
    )
    parser.add_argument(
        "--moves_limit",
        type=int,
        default=100,
        help="Maximum number of moves per game",
    )

    args = parser.parse_args()

    config = maz.get_config()
    config.channel_size = args.channel_size
    config.num_simulations = args.num_simulations
    config.batch_size = args.batch_size
    config.num_games = args.num_games
    config.num_threads = args.num_threads
    config.model_path = args.model_path
    config.training_file = args.training_file
    config.debug = args.debug

    print("Starting AlphaZero...")

    maz.init_globals()
    maz.init_model()

    print("Starting Evaluator thread.")
    maz.start_evaluator_thread()

    print(f"Starting worker.")
    maz.run_worker_threads()

    maz.stop_evaluator()


if __name__ == "__main__":
    main()
