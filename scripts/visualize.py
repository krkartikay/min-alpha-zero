import sys
import matplotlib
import torch
import chess
import chess.svg
import numpy as np
import matplotlib.pyplot as plt
import cairosvg


def board_to_tensor(board):
    """Convert chess board to tensor format matching chess_utils.hpp"""
    tensor = np.zeros((7, 8, 8), dtype=np.float32)

    # Convert piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            piece_type = piece.piece_type - 1  # 0-based indexing
            color = 1.0 if piece.color == chess.WHITE else -1.0
            tensor[piece_type, rank, file] = color

    # Add turn channel (channel 6)
    turn_value = 1.0 if board.turn == chess.WHITE else -1.0
    tensor[6, :, :] = turn_value

    return tensor


def move_to_int(move):
    """Convert move to integer format matching chess_utils.hpp"""
    return move.from_square * 64 + move.to_square


def index_to_coordinates(i):
    """Map policy index to coordinates in the 64x64 grid."""
    from_sq = i // 64
    to_sq = i % 64
    x_big = from_sq // 8
    y_big = from_sq % 8
    x_small = to_sq // 8
    y_small = to_sq % 8
    x_all = x_big * 8 + x_small
    y_all = y_big * 8 + y_small
    return x_all, y_all


def visualize():
    # Load the model
    model = "model.pt" if (len(sys.argv) != 2 or not sys.argv[1]) else sys.argv[1]
    model = torch.jit.load(model)
    model.eval()

    # Scholar's mate position - White to move and mate in one
    # Position after: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# (mate)
    # board = chess.Board()
    board = chess.Board(
        "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4"
    )

    print(f"Board position:\n{board}")
    print(f"Legal moves: {len(list(board.legal_moves))}")

    # Convert board to tensor
    board_tensor = board_to_tensor(board)
    input_tensor = torch.tensor(board_tensor).unsqueeze(0)  # Add batch dimension

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Get model predictions
    with torch.no_grad():
        policy_logits, value = model(input_tensor)

    print(f"Value prediction: {value.item():.4f}")

    # Convert policy to probabilities
    policy_probs = torch.softmax(policy_logits, dim=1)

    # Find best moves according to model
    legal_moves = list(board.legal_moves)
    legal_move_indices = [move_to_int(move) for move in legal_moves]

    # (1) Logits heatmap (unmasked)
    logits_np = policy_logits[0].detach().cpu().numpy()
    logits_map = np.zeros((64, 64), dtype=np.float32)
    for i in range(len(logits_np)):
        x, y = index_to_coordinates(i)
        logits_map[x, y] = float(logits_np[i])
    plt.figure(figsize=(10, 10))
    plt.imshow(logits_map, cmap="viridis", origin="lower")
    plt.colorbar(label="Policy Logits")
    plt.title("Policy Logits Heatmap (64x64 from-to moves)")
    plt.savefig("mate_in_one_logits_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Logits heatmap saved as 'mate_in_one_logits_heatmap.png'")

    # (2) Logits heatmap (masked to legal moves, illegal moves as -10)
    masked_logits_np = np.full_like(logits_np, -10.0)
    for idx in legal_move_indices:
        masked_logits_np[idx] = logits_np[idx]
    masked_logits_map = np.zeros((64, 64), dtype=np.float32)
    for i in range(len(masked_logits_np)):
        x, y = index_to_coordinates(i)
        masked_logits_map[x, y] = float(masked_logits_np[i])
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_logits_map, cmap="viridis", origin="lower")
    plt.colorbar(label="Masked Policy Logits")
    plt.title("Masked Policy Logits Heatmap (legal moves only)")
    plt.savefig("mate_in_one_masked_logits_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Masked logits heatmap saved as 'mate_in_one_masked_logits_heatmap.png'")

    # (3) Policy probability heatmap (softmax over masked logits, illegal moves as -1e10)
    masked_logits_for_softmax = np.full_like(logits_np, -1e10)
    for idx in legal_move_indices:
        masked_logits_for_softmax[idx] = logits_np[idx]
    masked_logits_tensor = torch.tensor(
        masked_logits_for_softmax, dtype=policy_logits.dtype
    )
    masked_policy_probs = torch.softmax(masked_logits_tensor, dim=0).cpu().numpy()
    masked_probs_map = np.zeros((64, 64), dtype=np.float32)
    for i in range(len(masked_policy_probs)):
        x, y = index_to_coordinates(i)
        masked_probs_map[x, y] = float(masked_policy_probs[i])
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_probs_map, cmap="viridis", origin="lower")
    plt.colorbar(label="Masked Policy Probability")
    plt.title("Masked Policy Probability Heatmap (softmax over masked logits)")
    plt.savefig(
        "mate_in_one_masked_policy_probs_heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(
        "Masked policy probability heatmap saved as 'mate_in_one_masked_policy_probs_heatmap.png'"
    )

    # Get move probabilities and sort
    move_probs = [
        (move, masked_policy_probs[move_to_int(move)].item()) for move in legal_moves
    ]
    move_probs.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 moves by model probability:")
    for i, (move, prob) in enumerate(move_probs[:10]):
        print(f"{i+1:2d}. {move}: {prob:.6f}")

    # Create board with arrows for top 20 moves
    arrows = []
    for i, (move, prob) in enumerate(move_probs[:20]):
        opacity = prob * 10  # opacity proportional to prob
        rgba_color = f"rgba(0, 60, 100, {opacity})"
        arrows.append(
            chess.svg.Arrow(move.from_square, move.to_square, color=rgba_color)
        )

    board_with_arrows_svg = chess.svg.board(board, arrows=arrows, size=400)

    # Save SVG file and convert to PNG
    with open("mate_in_one_board_with_arrows.svg", "w") as f:
        f.write(board_with_arrows_svg)

    png_data = cairosvg.svg2png(bytestring=board_with_arrows_svg.encode("utf-8"))
    with open("mate_in_one_board_with_arrows.png", "wb") as f:
        f.write(png_data)
    print(
        "Board with top 5 move arrows saved as 'mate_in_one_board_with_arrows.png' and .svg"
    )

    # Evaluate child positions using value head
    move_values = []
    child_tensors = []
    for move in legal_moves:
        child_board = board.copy()
        child_board.push(move)
        child_tensor = board_to_tensor(child_board)
        child_tensors.append(torch.tensor(child_tensor).unsqueeze(0).to(device))
    
    # Batch evaluate all child positions
    if child_tensors:
        child_batch = torch.cat(child_tensors, dim=0)
        with torch.no_grad():
            _, child_values = model(child_batch)
        child_values = child_values.squeeze()
        
        # For opponent's turn, negate values (what's good for them is bad for us)
        if not board.turn:  # If it was black's turn originally, children are white's turn
            child_values = -child_values
        
        move_values = list(zip(legal_moves, child_values.cpu().numpy()))
        move_values.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 moves by child position value:")
        for i, (move, value) in enumerate(move_values[:10]):
            print(f"{i+1:2d}. {move}: {value:.6f}")
        
        # Create board with arrows for top value moves
        value_arrows = []
        max_val = move_values[0][1]
        min_val = move_values[-1][1] if len(move_values) > 1 else max_val
        val_range = max_val - min_val if max_val != min_val else 1.0
        
        for i, (move, value) in enumerate(move_values[:20]):
            # Normalize opacity based on value ranking
            opacity = 0.7 * (value - min_val) / val_range
            rgba_color = f"rgba(100, 0, 60, {opacity})"
            value_arrows.append(
                chess.svg.Arrow(move.from_square, move.to_square, color=rgba_color)
            )
        
        board_with_value_arrows_svg = chess.svg.board(board, arrows=value_arrows, size=400)
        
        with open("mate_in_one_board_with_value_arrows.svg", "w") as f:
            f.write(board_with_value_arrows_svg)
        
        png_data = cairosvg.svg2png(bytestring=board_with_value_arrows_svg.encode("utf-8"))
        with open("mate_in_one_board_with_value_arrows.png", "wb") as f:
            f.write(png_data)
        print("Board with top value moves saved as 'mate_in_one_board_with_value_arrows.png' and .svg")

    # Check if the mate move (Qxf7#) is highly ranked
    mate_move = chess.Move.from_uci("h5f7")
    if mate_move in legal_moves:
        mate_idx = move_to_int(mate_move)
        mate_prob = policy_probs[0, mate_idx].item()
        print(f"\nMate move Qxf7# probability: {mate_prob:.4f}")

        # Check if it's the top move by policy
        best_move_idx = legal_move_indices[
            np.argmax([policy_probs[0, idx].item() for idx in legal_move_indices])
        ]
        best_move = legal_moves[legal_move_indices.index(best_move_idx)]
        print(f"Model's top choice (policy): {best_move}")

        if best_move == mate_move:
            print("SUCCESS: Model correctly identifies the mate in one (policy)!")
        else:
            print("Model did not identify the mate move as best (policy)")
        
        # Check value head ranking
        if move_values:
            mate_value = next((val for move, val in move_values if move == mate_move), None)
            best_value_move = move_values[0][0]
            print(f"Mate move value: {mate_value:.6f}")
            print(f"Model's top choice (value): {best_value_move}")
            
            if best_value_move == mate_move:
                print("SUCCESS: Model correctly identifies the mate in one (value)!")
            else:
                print("Model did not identify the mate move as best (value)")
    else:
        print("ERROR: Mate move not in legal moves")


if __name__ == "__main__":
    visualize()
