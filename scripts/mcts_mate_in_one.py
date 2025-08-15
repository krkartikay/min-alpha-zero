#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import min_alpha_zero as maz

def print_game_tree(node, depth=0, max_depth=3, action=-1):
    if depth > max_depth:
        return
    
    indent = "  " * depth
    history_str = f" ({node.move_history})" if node.move_history else ""
    visits = node.child_visits
    visits_total = sum(visits)
    
    print(f"{indent}Node{history_str}: visits={visits_total}, value={node.value:.3f}, evaluated={node.is_evaluated}, leaf={node.is_leaf}")
    
    if depth < max_depth and visits_total > 0:
        for i in range(maz.NUM_ACTIONS):
            if visits[i] > 0:
                child = node.get_child_node(i)
                if child:
                    print_game_tree(child, depth + 1, max_depth, i)

def main():
    config = maz.get_config()
    config.channel_size = 16
    config.num_simulations = 200  # More simulations for mate-in-one
    config.batch_size = 10
    config.model_path = "model.pt"
    config.debug = False
    
    print("Initializing globals...")
    maz.init_globals()
    
    print("Initializing model...")
    maz.init_model()
    
    print("Starting evaluator thread...")
    maz.start_evaluator_thread()
    
    # Scholar's mate position - White to move and mate in one
    # Position after: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# (mate)
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4"
    
    print(f"Creating game from mate-in-one position...")
    print(f"FEN: {fen}")
    
    # Create a game from FEN string
    game = maz.Game(fen)
    root = game.get_root()
    
    print(f"Initial board:\n{maz.board_to_string(root.board)}")
    
    # Check legal moves and if mate move is legal
    legal_mask = root.legal_mask
    mate_action = 39 * 64 + 53  # h5 = 39, f7 = 53
    print(f"\nMate move h5->f7 (action {mate_action}) is legal: {legal_mask[mate_action]}")
    
    # Print some legal moves for reference
    legal_actions = [i for i in range(len(legal_mask)) if legal_mask[i]]
    print(f"Total legal moves: {len(legal_actions)}")
    print("First 10 legal actions:", legal_actions[:10])
    if mate_action in legal_actions:
        print(f"Mate action {mate_action} is in legal actions at index {legal_actions.index(mate_action)}")
    
    print(f"\nRunning {config.num_simulations} MCTS simulations...")
    for _ in range(config.num_simulations):
        game.run_simulation()
    
    print("\nMCTS tree structure:")
    print_game_tree(root)
    
    best_move = game.select_move()
    print(f"\nBest move selected: {best_move}")
    
    print("\nChild visit counts (top 10):")
    visits = root.child_visits
    visit_pairs = [(i, visits[i]) for i in range(maz.NUM_ACTIONS) if visits[i] > 0]
    visit_pairs.sort(key=lambda x: x[1], reverse=True)
    for action, visit_count in visit_pairs[:10]:
        from_sq = action // 64
        to_sq = action % 64
        print(f"  Move {from_sq}->{to_sq}: {visit_count} visits")
    
    # Check if MCTS found the mate move (h5f7, which is 39*64 + 53 = 2549)
    mate_action = 39 * 64 + 53  # h5 = 39, f7 = 53
    mate_visits = visits[mate_action]
    print(f"\nMate move h5->f7 (action {mate_action}): {mate_visits} visits")
    
    if mate_action == best_move:
        print("SUCCESS: MCTS found the mate in one!")
    else:
        print("MCTS did not select the mate move as best")
    
    print("\nStopping evaluator for clean shutdown...")
    maz.stop_evaluator()

if __name__ == "__main__":
    main()
