#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import min_alpha_zero as maz

def print_game_tree(node, depth=0, max_depth=3, action=-1):
    if depth > max_depth:
        return
    
    indent = "  " * depth
    action_str = f" (action {action})" if action >= 0 else ""
    visits = node.child_visits
    visits_total = sum(visits)
    
    print(f"{indent}Node{action_str}: visits={visits_total}, value={node.value:.3f}, evaluated={node.is_evaluated}, leaf={node.is_leaf}")
    
    if depth < max_depth and visits_total > 0:
        for i in range(maz.NUM_ACTIONS):
            if visits[i] > 0:
                child = node.get_child_node(i)
                if child:
                    print_game_tree(child, depth + 1, max_depth, i)

def main():
    config = maz.get_config()
    config.channel_size = 16
    config.num_simulations = 50
    config.batch_size = 10
    config.model_path = "model.pt"
    config.debug = False
    
    print("Initializing globals...")
    maz.init_globals()
    
    print("Initializing model...")
    maz.init_model()
    
    print("Starting evaluator thread...")
    maz.start_evaluator_thread()
    
    print("Creating game and running MCTS search...")
    game = maz.Game()
    root = game.get_root()
    
    print(f"Initial board:\n{maz.board_to_string(root.board)}")
    
    print(f"Running {config.num_simulations} MCTS simulations...")
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
    
    print("\nStopping evaluator for clean shutdown...")
    maz.stop_evaluator()

if __name__ == "__main__":
    main()