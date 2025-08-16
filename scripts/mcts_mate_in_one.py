#!/usr/bin/env python3

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

def run_single_test(num_simulations, mate_action):
    """Run a single MCTS test and return whether mate move was selected and its probability"""
    game = maz.Game("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    
    # select_move() already runs num_simulations internally
    best_move = game.select_move()
    
    root = game.get_root()
    visits = root.child_visits
    total_visits = sum(visits)
    mate_visits = visits[mate_action]
    mate_probability = mate_visits / total_visits if total_visits > 0 else 0.0
    
    return mate_action == best_move, mate_probability

def main():
    config = maz.get_config()
    config.channel_size = 16
    config.num_simulations = 100
    config.batch_size = 10
    config.model_path = "model.pt"
    config.debug = False
    config.temperature = 0.1
    
    num_tests = 100
    
    print("Initializing globals...")
    maz.init_globals()
    
    print("Initializing model...")
    maz.init_model()
    
    print("Starting evaluator thread...")
    maz.start_evaluator_thread()
    
    # Scholar's mate position - White to move and mate in one
    # Position after: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7# (mate)
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4"
    mate_action = 39 * 64 + 53  # h5 = 39, f7 = 53
    
    print(f"FEN: {fen}")
    print(f"Mate move h5->f7 (action {mate_action})")
    
    # Run multiple tests
    successes = 0
    
    print(f"\nRunning {num_tests} tests with {config.num_simulations} simulations each...")
    
    for test_num in range(1, num_tests + 1):
        success, probability = run_single_test(config.num_simulations, mate_action)
        successes += success
        status = "SUCCESS" if success else "FAILED"
        print(f"Test {test_num}: {status} (mate move probability: {probability:.3f})")
    
    success_rate = successes / num_tests
    print(f"\nResults: {successes}/{num_tests} tests found the mate move")
    print(f"Success rate: {success_rate:.1%}")
    
    print("\nStopping evaluator for clean shutdown...")
    maz.stop_evaluator()

if __name__ == "__main__":
    main()
