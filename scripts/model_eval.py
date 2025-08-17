#!/usr/bin/env python3
import argparse
import min_alpha_zero as maz


def main():
    parser = argparse.ArgumentParser(description="Model evaluation against agents")
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
        default=100,
        help="Number of games to play in evaluation",
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
        help="Enable debug logging and intermediate state dumps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter for move selection",
    )

    args = parser.parse_args()

    config = maz.get_config()
    config.channel_size = args.channel_size
    config.num_simulations = args.num_simulations
    config.batch_size = args.batch_size
    config.num_games = args.num_games
    config.model_path = args.model_path
    config.training_file = args.training_file
    config.debug = args.debug
    config.temperature = args.temperature

    print("Starting AlphaZero agent tournament...")

    maz.init_globals()
    maz.init_model()

    maz.start_evaluator_thread()

    agent1 = maz.RawModelAgent()
    agent2 = maz.RandomAgent()
    maz.run_agent_tournament(agent1, agent2)

    maz.stop_evaluator()


if __name__ == "__main__":
    main()
