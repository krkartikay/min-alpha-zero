#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np

def parse_selfplay_log(log_file):
    """Parse selfplay_games.log and extract games with their moves."""
    games = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract game number and moves
            match = re.match(r'.*Game (\d+): (.+)', line)
            if match:
                game_num = int(match.group(1))
                moves_str = match.group(2)
                
                # Extract first two moves (4 plies)
                moves = re.findall(r'\d+\.\s*([a-h]?\w+)\s*([a-h]?\w+)?', moves_str)
                if moves:
                    first_move = moves[0][0] if moves[0][0] else None
                    second_move = moves[0][1] if len(moves[0]) > 1 and moves[0][1] else None
                    
                    games.append({
                        'game_num': game_num,
                        'first_move': first_move,
                        'second_move': second_move,
                        'opening': f"{first_move} {second_move}" if first_move and second_move else first_move
                    })
    
    return games

def analyze_opening_frequencies(games, window_size=100):
    """Analyze opening frequencies over time with sliding window."""
    first_move_counts = []
    second_move_counts = []
    opening_counts = []
    game_indices = []
    
    for i in range(len(games)):
        start_idx = max(0, i - window_size + 1)
        window_games = games[start_idx:i+1]
        
        # Count first moves
        first_moves = Counter(g['first_move'] for g in window_games if g['first_move'])
        
        # Count openings (first two moves)
        openings = Counter(g['opening'] for g in window_games if g['opening'] and ' ' in g['opening'])
        
        first_move_counts.append(dict(first_moves))
        opening_counts.append(dict(openings))
        game_indices.append(i)
    
    return game_indices, first_move_counts, opening_counts

def plot_opening_frequencies(log_file='selfplay_games.log', window_size=100, top_n=8):
    """Create plots showing opening frequencies over time."""
    print(f"Parsing {log_file}...")
    games = parse_selfplay_log(log_file)
    print(f"Found {len(games)} games")
    
    if len(games) < window_size:
        print(f"Not enough games for window size {window_size}")
        return
    
    game_indices, first_move_counts, opening_counts = analyze_opening_frequencies(games, window_size)
    
    # Get most common first moves and openings
    all_first_moves = Counter(g['first_move'] for g in games if g['first_move'])
    all_openings = Counter(g['opening'] for g in games if g['opening'] and ' ' in g['opening'])
    
    top_first_moves = [move for move, _ in all_first_moves.most_common(top_n)]
    top_openings = [opening for opening, _ in all_openings.most_common(top_n)]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: First move frequencies
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(top_first_moves)))
    
    for move, color in zip(top_first_moves, colors1):
        frequencies = []
        for counts in first_move_counts:
            total = sum(counts.values())
            freq = counts.get(move, 0) / total if total > 0 else 0
            frequencies.append(freq * 100)  # Convert to percentage
        
        ax1.plot(game_indices, frequencies, label=move, color=color, linewidth=2)
    
    ax1.set_xlabel('Game Number')
    ax1.set_ylabel('Frequency (%)')
    ax1.set_title(f'First Move Frequencies Over Time (Window: {window_size} games)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Opening (first 2 moves) frequencies  
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(top_openings)))
    
    for opening, color in zip(top_openings, colors2):
        frequencies = []
        for counts in opening_counts:
            total = sum(counts.values())
            freq = counts.get(opening, 0) / total if total > 0 else 0
            frequencies.append(freq * 100)  # Convert to percentage
        
        ax2.plot(game_indices, frequencies, label=opening, color=color, linewidth=2)
    
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Frequency (%)')
    ax2.set_title(f'Opening (First 2 Moves) Frequencies Over Time (Window: {window_size} games)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('opening_frequencies.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nMost common first moves (total {len(games)} games):")
    for move, count in all_first_moves.most_common(10):
        print(f"  {move}: {count} ({count/len(games)*100:.1f}%)")
    
    print(f"\nMost common openings:")
    for opening, count in all_openings.most_common(10):
        print(f"  {opening}: {count} ({count/len([g for g in games if g['opening'] and ' ' in g['opening']])*100:.1f}%)")

if __name__ == "__main__":
    plot_opening_frequencies()