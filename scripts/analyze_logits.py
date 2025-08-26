#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import chess
import sys
import os

sys.path.append(".")
import min_alpha_zero


def get_scholar_mate_position():
    """Get the scholar's mate position"""
    board = chess.Board()
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def board_to_tensor(board):
    """Convert chess board to tensor"""
    tensor = torch.zeros(7, 8, 8)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            channel = piece_map[piece.piece_type]
            value = 1.0 if piece.color == chess.WHITE else -1.0
            tensor[channel, rank, file] = value

    # Turn channel
    tensor[6, 0, 0] = 1.0 if board.turn == chess.WHITE else -1.0

    return tensor


def analyze_model_logits(model_path):
    """Analyze logits distribution for a model"""
    print(f"Analyzing {model_path}...")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # Get position
    board = get_scholar_mate_position()
    board_tensor = (
        board_to_tensor(board).unsqueeze(0).to(device)
    )  # Add batch dimension and move to device

    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        policy_logits = policy_logits.squeeze(
            0
        ).cpu()  # Remove batch dimension and move to CPU

    # Get legal moves mask
    legal_actions = []
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        action = from_sq * 64 + to_sq
        legal_actions.append(action)

    # Mate move action (h5 to f7)
    mate_action = chess.H5 * 64 + chess.F7

    return {
        "model_path": model_path,
        "all_logits": policy_logits.numpy(),
        "legal_logits": policy_logits[legal_actions].numpy(),
        "mate_logit": policy_logits[mate_action].item(),
        "legal_actions": legal_actions,
        "mate_action": mate_action,
    }


def plot_logits_comparison():
    """Plot logits histogram comparison"""
    # models = ["model.pt"]  # Current trained model
    models = []  # Current trained model

    # Add all models in out/ directory
    if os.path.exists("out"):
        for f in sorted(os.listdir("out")):
            if f.endswith(".pt"):
                models.append(f"out/{f}")

    models = models[-3:]  # Keep only the latest 3 models

    results = []
    for model_path in models:
        if os.path.exists(model_path):
            try:
                result = analyze_model_logits(model_path)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {model_path}: {e}")

    if not results:
        print("No models found to analyze")
        return

    # Create subplots
    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        model_name = os.path.basename(result["model_path"])

        # Plot all logits histogram
        axes[0, i].hist(
            result["all_logits"], bins=50, alpha=0.7, color="blue", label="All actions"
        )
        axes[0, i].axvline(
            result["mate_logit"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Mate move: {result["mate_logit"]:.3f}',
        )
        axes[0, i].set_title(f"{model_name}\nAll Logits")
        axes[0, i].set_xlabel("Logit value")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Plot legal logits histogram
        axes[1, i].hist(
            result["legal_logits"],
            bins=20,
            alpha=0.7,
            color="green",
            label="Legal actions",
        )
        axes[1, i].axvline(
            result["mate_logit"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Mate move: {result["mate_logit"]:.3f}',
        )
        axes[1, i].set_title(f"{model_name}\nLegal Logits")
        axes[1, i].set_xlabel("Logit value")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

        # Print stats
        print(f"\n{model_name}:")
        print(
            f"  All logits - mean: {np.mean(result['all_logits']):.3f}, std: {np.std(result['all_logits']):.3f}"
        )
        print(
            f"  Legal logits - mean: {np.mean(result['legal_logits']):.3f}, std: {np.std(result['legal_logits']):.3f}"
        )
        print(f"  Mate logit: {result['mate_logit']:.6f}")
        print(
            f"  Mate rank among legal: {np.sum(result['legal_logits'] > result['mate_logit']) + 1}/{len(result['legal_logits'])}"
        )

    plt.tight_layout()
    plt.savefig("logits_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to logits_analysis.png")

    # Also save softmax probabilities comparison
    fig2, axes2 = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes2 = [axes2]

    for i, result in enumerate(results):
        model_name = os.path.basename(result["model_path"])

        # Convert legal logits to probabilities
        legal_logits = torch.tensor(result["legal_logits"])
        legal_probs = torch.softmax(legal_logits, dim=0).numpy()
        mate_idx = np.where(np.array(result["legal_actions"]) == result["mate_action"])[
            0
        ]
        mate_prob = legal_probs[mate_idx[0]] if len(mate_idx) > 0 else 0

        axes2[i].hist(
            legal_probs, bins=20, alpha=0.7, color="purple", label="Legal move probs"
        )
        axes2[i].axvline(
            mate_prob,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mate prob: {mate_prob:.6f}",
        )
        axes2[i].set_title(f"{model_name}\nLegal Move Probabilities")
        axes2[i].set_xlabel("Probability")
        axes2[i].set_ylabel("Frequency")
        axes2[i].legend()
        axes2[i].grid(True, alpha=0.3)

        print(f"  Mate probability: {mate_prob:.6f}")

    plt.tight_layout()
    plt.savefig("probabilities_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Probability plot saved to probabilities_analysis.png")


if __name__ == "__main__":
    plot_logits_comparison()
