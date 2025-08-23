#!/usr/bin/env python3

import numpy as np
import torch
from dataset import TrainingDataset
import chess


def tensor_to_board(board_tensor):
    """Convert 7x8x8 tensor back to chess board"""
    board = chess.Board(fen=None)
    board.clear()

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    # Decode pieces
    for channel in range(6):
        for rank in range(8):
            for file in range(8):
                value = board_tensor[channel, rank, file].item()
                if value != 0:
                    square = chess.square(file, rank)
                    piece_type = piece_types[channel]
                    color = chess.WHITE if value > 0 else chess.BLACK
                    piece = chess.Piece(piece_type, color)
                    board.set_piece_at(square, piece)

    # Set turn
    turn_value = board_tensor[6, 0, 0].item()
    board.turn = chess.WHITE if turn_value > 0 else chess.BLACK

    return board


def action_to_san(action, board):
    """Convert action index to SAN notation"""
    try:
        from_sq = action // 64
        to_sq = action % 64
        from_square = chess.Square(from_sq)
        to_square = chess.Square(to_sq)

        # Try to create a move
        move = chess.Move(from_square, to_square)

        # Check for promotions
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_sq >= 56) or (
                piece.color == chess.BLACK and to_sq <= 7
            ):
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

        # Check if move is legal
        if move in board.legal_moves:
            return board.san(move)
        else:
            # Try different promotions
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                if move in board.legal_moves:
                    return board.san(move)

            # If still not legal, return algebraic notation
            return chess.square_name(from_sq) + chess.square_name(to_sq)
    except:
        return f"action{action}"


def print_training_records(start_record=0, num_records=10):
    print("Loading training data...")
    dataset = TrainingDataset("training_data.bin")
    print(f"Total records: {len(dataset)}")

    end_record = min(start_record + num_records, len(dataset))

    for i in range(start_record, end_record):
        record = dataset[i]
        board_tensor = record["board_tensor"]
        final_value = record["final_value"].item()
        visits = record["child_visit_counts"]
        child_values = record["child_values"]
        value = record["value"]
        policy = record["policy"]
        legal_mask = record["legal_mask"]

        total_visits = visits.sum().item()

        try:
            board = tensor_to_board(board_tensor)

            print(f"\n{'='*60}")
            print(f"RECORD {i}")
            print(f"Final: {final_value}, Total visits: {total_visits}")
            print(f"Turn: {'White' if board.turn else 'Black'}")
            print(f"{'='*60}")
            print(board)

            # Show top 3 moves by visit counts
            legal_indices = [j for j in range(len(legal_mask)) if legal_mask[j]]
            if len(legal_indices) > 0:
                legal_visits = [(j, visits[j].item()) for j in legal_indices]
                legal_visits.sort(key=lambda x: x[1], reverse=True)

                print("Top 3 moves by visit counts:")
                for rank, (action, visit_count) in enumerate(legal_visits[:3]):
                    san_move = action_to_san(action, board)
                    prob = policy[action].item()
                    child_val = child_values[action].item()
                    print(
                        f"  {rank+1}. {san_move}: visits={visit_count}, policy={prob:.6f}, child_value={child_val:.6f}"
                    )
            else:
                print("No legal moves found")

        except Exception as e:
            print(f"Record {i}: Error decoding: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        start = int(sys.argv[1])
        count = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
        print_training_records(start, count)
    else:
        print("Usage: python print_training_records.py <start_record> [count]")
        print("Example: python print_training_records.py 0 5")
        print_training_records(0, 5)
