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

def test_model(model_path):
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path}")
    print('='*60)
    
    config = maz.get_config()
    config.model_path = model_path
    
    print("Initializing model...")
    maz.init_model()
    
    # Scholar's mate position - White to move and mate in one
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4"
    mate_action = 39 * 64 + 53  # h5 = 39, f7 = 53
    
    # Run single test
    game = maz.Game(fen)
    best_move = game.select_move()
    
    root = game.get_root()
    visits = root.child_visits
    policy = root.policy
    legal_mask = root.legal_mask
    total_visits = sum(visits)
    mate_visits = visits[mate_action]
    mate_probability = mate_visits / total_visits if total_visits > 0 else 0.0
    
    success = mate_action == best_move
    status = "SUCCESS" if success else "FAILED"
    
    print(f"Result: {status}")
    print(f"Mate move policy: {policy[mate_action]:.6f}")
    print(f"Mate move visits: {mate_visits}")
    print(f"Total visits: {total_visits}")
    
    # Find top 5 policy moves
    legal_actions = [i for i in range(len(legal_mask)) if legal_mask[i]]
    legal_policies = [(i, policy[i]) for i in legal_actions]
    legal_policies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 5 policy moves:")
    for i, (action, prob) in enumerate(legal_policies[:5]):
        from_sq = action // 64
        to_sq = action % 64
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file, to_rank = to_sq % 8, to_sq // 8
        files = "abcdefgh"
        move_str = f"{files[from_file]}{from_rank+1}{files[to_file]}{to_rank+1}"
        is_mate = "(MATE!)" if action == mate_action else ""
        print(f"  {i+1}. {move_str} - policy={prob:.6f} {is_mate}")
    
    return success, policy[mate_action], mate_visits

def main():
    # Set up config first
    config = maz.get_config()
    config.channel_size = 16  # Power of 2
    config.num_simulations = 100
    config.batch_size = 10
    config.debug = False
    config.temperature = 0.1
    
    print("Initializing globals...")
    maz.init_globals()
    
    print("Starting evaluator thread...")
    maz.start_evaluator_thread()
    
    # Test all models in out/ directory
    models = ["model.pt", "out/model_001.pt", "out/model_002.pt", "out/model_003.pt", "out/model_004.pt"]
    
    results = []
    for model_path in models:
        try:
            success, mate_policy, mate_visits = test_model(model_path)
            results.append((model_path, success, mate_policy, mate_visits))
        except Exception as e:
            print(f"Error testing {model_path}: {e}")
            results.append((model_path, False, 0.0, 0))
    
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL MODELS")
    print('='*60)
    print(f"{'Model':<20} {'Success':<8} {'Policy':<10} {'Visits':<7}")
    print('-'*50)
    for model_path, success, mate_policy, mate_visits in results:
        model_name = model_path.split('/')[-1]
        status = "SUCCESS" if success else "FAILED"
        print(f"{model_name:<20} {status:<8} {mate_policy:<10.6f} {mate_visits:<7}")
    
    print("\nStopping evaluator for clean shutdown...")
    maz.stop_evaluator()

if __name__ == "__main__":
    main()