import sys
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

def test_mate_in_one():
    # Load the model
    model = "model.pt" if (len(sys.argv) != 2 or not sys.argv[1]) else sys.argv[1]
    model = torch.jit.load(model)
    model.eval()

    # Scholar's mate position - White to move and mate in one
    # Position after: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# (mate)
    board = chess.Board()
    # board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    
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
    
    # Get move probabilities and sort
    move_probs = [(move, policy_probs[0, move_to_int(move)].item()) for move in legal_moves]
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 moves by model probability:")
    for i, (move, prob) in enumerate(move_probs[:10]):
        print(f"{i+1:2d}. {move}: {prob:.6f}")
    
    # Create policy heatmap with matplotlib
    policy_heatmap_data = np.zeros((64, 64))
    masked_policy_heatmap_data = np.zeros((64, 64))
    for i, policy in enumerate(policy_probs[0]):
        x, y = index_to_coordinates(i)
        policy_heatmap_data[x][y] = policy.item()
        if i in legal_move_indices:
            masked_policy_heatmap_data[x][y] = policy.item()
        else:
            masked_policy_heatmap_data[x][y] = 0.0

    plt.figure(figsize=(10, 10))
    plt.imshow(policy_heatmap_data, cmap='viridis', origin='lower')
    plt.colorbar(label='Policy Probability')
    plt.title('Policy Heatmap (64x64 from-to moves)')
    plt.xlabel('To Square Y')
    plt.ylabel('From Square X')
    plt.savefig('mate_in_one_policy_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPolicy heatmap saved as 'mate_in_one_policy_heatmap.png'")
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_policy_heatmap_data, cmap='viridis', origin='lower')
    plt.colorbar(label='Masked Policy Probability')
    plt.title('Policy Heatmap (64x64 from-to moves)')
    plt.xlabel('To Square Y')
    plt.ylabel('From Square X')
    plt.savefig('mate_in_one_masked_policy_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPolicy heatmap saved as 'mate_in_one_masked_policy_heatmap.png'")

    # Create board with arrows for top 5 moves
    arrows = []
    for i, (move, prob) in enumerate(move_probs[:5]):
        color = ["red", "orange", "yellow", "green", "blue"][i]
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))
    
    board_with_arrows_svg = chess.svg.board(board, arrows=arrows, size=400)
    
    # Save SVG file and convert to PNG
    with open('mate_in_one_board_with_arrows.svg', 'w') as f:
        f.write(board_with_arrows_svg)
    
    png_data = cairosvg.svg2png(bytestring=board_with_arrows_svg.encode('utf-8'))
    with open('mate_in_one_board_with_arrows.png', 'wb') as f:
        f.write(png_data)
    print("Board with top 5 move arrows saved as 'mate_in_one_board_with_arrows.png' and .svg")
    
    # Check if the mate move (Qxf7#) is highly ranked
    mate_move = chess.Move.from_uci("h5f7")
    if mate_move in legal_moves:
        mate_idx = move_to_int(mate_move)
        mate_prob = policy_probs[0, mate_idx].item()
        print(f"\nMate move Qxf7# probability: {mate_prob:.4f}")
        
        # Check if it's the top move
        best_move_idx = legal_move_indices[np.argmax([policy_probs[0, idx].item() for idx in legal_move_indices])]
        best_move = legal_moves[legal_move_indices.index(best_move_idx)]
        print(f"Model's top choice: {best_move}")
        
        if best_move == mate_move:
            print("SUCCESS: Model correctly identifies the mate in one!")
        else:
            print("Model did not identify the mate move as best")
    else:
        print("ERROR: Mate move not in legal moves")

if __name__ == "__main__":
    test_mate_in_one()