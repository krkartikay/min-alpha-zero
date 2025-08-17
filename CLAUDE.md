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
- Primary output: `min_alpha_zero` Python module with C++ bindings

## Training Pipeline (Python-Centric)

```
python scripts/run_all.py              # Complete training loop (infinite)
python scripts/alpha_zero.py           # Self-play generation
python scripts/train.py                # Neural network training
python scripts/model_eval.py           # Agent tournament evaluation
python scripts/model.py                # Create initial model
```

- **Python-first workflow**: All orchestration through Python scripts using C++ bindings
- Training data: binary format in `training_data.bin` (see scripts/dataset.py:25-37)
- Models: TorchScript `.pt` files, versioned in `out/model_NNN.pt`
- Continuous training: `run_all.py` provides infinite model→self-play→training→evaluation cycle

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

## Analysis and Debugging Tools

```
python scripts/mcts_demo.py              # Interactive MCTS tree visualization
python scripts/debug_mcts.py             # Multi-model MCTS testing
python scripts/mcts_mate_in_one.py       # Tactical position testing
python scripts/analyze_logits.py         # Model behavior analysis
python scripts/print_training_records.py # Training data inspection
python scripts/dashboards/explore_data.py # Interactive data explorer (Dash)
python scripts/dashboards/logs.py        # Real-time log monitoring (Dash)
```

- **Advanced visualization**: Interactive dashboards with chess board SVG rendering
- **Policy analysis**: Move probability heatmaps and top-move arrows
- **Training monitoring**: Real-time log updates and data exploration
- **Tactical testing**: Specialized mate-in-one position evaluation

## File Structure

```
include/
├── alpha_zero.h      # Core MCTS/NN classes and functions  
├── chess.hpp         # External chess library (5195 lines)
└── chess_utils.hpp   # Board↔tensor, move↔int conversion

src/
├── agents.cpp        # ChessAgent implementations + tournament system
├── evaluator.cpp     # Batched NN evaluation fiber with CUDA
├── worker.cpp        # MCTS simulation and self-play logic
├── logging.cpp       # Custom Abseil log sink + tree visualization
└── python_bindings.cpp # Comprehensive pybind11 module "min_alpha_zero"

scripts/                # Python package for training and analysis
├── alpha_zero.py     # Self-play execution with C++ bindings
├── model.py          # 8-layer ResNet with policy/value heads
├── train.py          # Model training with masked softmax
├── dataset.py        # Binary training data management
├── run_all.py        # Complete training orchestration
├── model_eval.py     # Agent tournament evaluation
├── analyze_logits.py # Model behavior analysis
├── debug_mcts.py     # MCTS debugging and visualization
├── mcts_demo.py      # Interactive MCTS demonstration
├── mcts_mate_in_one.py # Tactical position testing
├── print_training_records.py # Training data inspection
├── dashboards/       # Interactive web-based analysis tools
│   ├── explore_data.py # Training data explorer with chess visualization
│   └── logs.py       # Real-time log monitoring dashboard
└── test/            # Python-based testing and validation
    ├── mate_in_one.py # Pure Python chess puzzle testing
    └── test_board.py # Basic functionality validation

test/
└── test_all.cpp      # Comprehensive C++ test suite (Google Test)

backlog/
└── todo_001.md       # Spatial move encoding architecture proposal

external/             # Git submodules
├── abseil-cpp/       # Google utilities, logging, flags
└── pybind11/         # Python binding generation
```

## Testing

```
./build/test_all                          # C++ unit tests (Google Test)
python scripts/test/mate_in_one.py        # Python chess puzzle tests
python scripts/test/test_board.py         # Basic board functionality
python scripts/model_eval.py              # Agent tournament evaluation
```

- **C++ tests**: Comprehensive unit tests for Node, MCTS, chess utilities, and agents
- **Python tests**: End-to-end model validation and tactical position testing
- **Integration tests**: Multi-model comparison and statistical validation
- **No lint commands configured**: Code follows C++17 standards with strict warnings

## Key Improvements and Architecture Changes

### Python-Centric Workflow
- **Migration**: From C++ executables to Python orchestration with C++ bindings
- **Benefit**: Enhanced experimentation, visualization, and data analysis capabilities
- **Integration**: Seamless numpy array integration for efficient data transfer

### Advanced Visualization System  
- **Interactive dashboards**: Dash-based real-time training data exploration
- **Chess visualization**: SVG board rendering with policy heatmaps and move arrows
- **Monitoring**: Live log updates and training progress tracking

### Comprehensive Analysis Tools
- **Model comparison**: Multi-iteration performance evaluation
- **Tactical testing**: Specialized mate-in-one position validation
- **Debug capabilities**: MCTS tree visualization and model behavior analysis
- **Training insights**: Detailed data inspection and policy analysis

### Performance Optimizations
- **Cooperative threading**: Boost.Fiber for MCTS parallelization
- **GPU acceleration**: Batched CUDA neural network evaluation
- **Memory efficiency**: Lazy child node initialization and optimized tensor management

### Future Architecture (Backlog)
- **Spatial move encoding**: Proposed 2D convolutional policy head architecture
- **Enhanced patterns**: Leveraging spatial relationships in move representation
- **Breaking changes**: Would require complete model retraining