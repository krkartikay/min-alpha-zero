#!/usr/bin/env python3

import numpy as np
import torch
from dataset import TrainingDataset
import chess

from chess_utils import tensor_to_board, action_to_san


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
