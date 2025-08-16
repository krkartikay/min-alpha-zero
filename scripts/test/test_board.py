#!/usr/bin/env python3
import min_alpha_zero as maz

def main():
    before_castle = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    
    # Use Node instead of Board since Board() doesn't take FEN
    node = maz.Node(before_castle)
    legal_mask = node.legal_mask
    
    # Print legal moves using the legal mask
    for i in range(maz.NUM_ACTIONS):
        if legal_mask[i]:
            from_square = i // 64
            to_square = i % 64
            print(f"Move {i}: From: {from_square} To: {to_square}")

if __name__ == '__main__':
    main()