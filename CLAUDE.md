# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

```
cmake --preset build
cmake --build build --parallel
```

- Uses CMake 3.20+ with C++17
- Dependencies: LibTorch, Boost (fiber/context), GTest, Abseil, pybind11
- CMakePresets.json defines "build" preset with hardcoded torch paths
- Builds to `build/` directory

## Key Executables

```
./build/alpha_zero [--num_games=N] [--num_simulations=N] [--batch_size=N]
./build/test_all           # GTest test suite
./build/model_eval         # Model evaluation against agents
./build/test_board         # Board functionality test
```

## Training Pipeline

```
./run.sh [--num_iterations=N] [--num_games=X]  # Full training loop
python model.py                                # Create initial model
python train.py                                # Train on existing data
python -c "import min_alpha_zero; ..."         # Python bindings
```

- `run.sh`: Orchestrates model→self-play→training→evaluation cycle
- Training data: binary format in `training_data.bin` (see dataset.py:8-15)
- Models: TorchScript `.pt` files, versioned in `out/model_NNN.pt`

## Architecture

### Core Classes
- `alphazero::Node`: MCTS tree node (alpha_zero.h:69-106)
  - `board`: chess::Board state
  - `policy[4096]`: move priors from NN
  - `child_visits[4096]`/`child_values[4096]`: MCTS stats
  - `legal_mask[4096]`: valid moves bitmap
- `alphazero::Game`: Self-play game manager (alpha_zero.h:125-140)
  - `root`: current MCTS tree root
  - `history`: GameState vector for training
- `alphazero::Config`: Runtime parameters (alpha_zero.h:52-63)

### Threading Model
- Boost.Fiber cooperative threading
- Worker fibers: MCTS simulation (worker.cpp)
- Evaluator fiber: batched NN inference (evaluator.cpp)
- `g_evaluation_queue`: buffered_channel for eval requests

### Move Encoding
- Actions: 64×64 = 4096 (from_square×64 + to_square)
- Board tensor: 7×8×8 (6 piece types + turn channel)
- chess_utils.hpp: move_to_int/int_to_move conversion

### Agents Framework
- `ChessAgent` base class (alpha_zero.h:157-162)
- `RandomAgent`, `MCTSAgent` implementations
- Tournament evaluation in agents.cpp

## File Structure

```
include/
├── alpha_zero.h      # Core MCTS/NN classes and functions
├── chess.hpp         # External chess library (large)
└── chess_utils.hpp   # Board↔tensor, move↔int conversion

src/
├── main.cpp          # CLI entry point with absl flags
├── agents.cpp        # ChessAgent implementations + tournament
├── evaluator.cpp     # Batched NN evaluation fiber
├── worker.cpp        # MCTS simulation logic
├── logging.cpp       # FileSink for absl::log
├── python_bindings.cpp # pybind11 module "min_alpha_zero"
└── model_eval.cpp    # Standalone model evaluation

test/
├── test_all.cpp      # GTest suite for core functionality
├── mate_in_one.py    # Chess puzzle visualization
└── *.png/*.svg       # Generated visualizations

dashboards/
├── explore_data.py   # Training data analysis
├── logs.py          # Log file parsing/visualization
└── requirements.txt  # matplotlib, pandas

External: abseil-cpp/, pybind11/ (git submodules)
```

## Testing

```
./build/test_all                    # Run all C++ tests
python test/mate_in_one.py          # Chess puzzle test
python dashboards/explore_data.py   # Data analysis
```

No lint commands configured. Tests validate Node construction, MCTS operations, move encoding, and game mechanics.