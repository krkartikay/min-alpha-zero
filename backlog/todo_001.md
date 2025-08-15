# TODO 001: Spatial Move Encoding Architecture

## Problem Statement
Current move encoding uses a linear mapping (`from * 64 + to`) that doesn't leverage the spatial relationships inherent in the 64×64 move representation grid. The policy head uses a fully connected layer that treats each move independently.

## Proposed Solution
Restructure the architecture to use spatial move encoding with convolutional policy head:

### 1. Update Move Encoding (`chess_utils.hpp`)
Replace linear encoding with spatial coordinate mapping:
```cpp
// Current: return from * 64 + to
// New: Map to spatial coordinates matching heatmap visualization
int move_to_int(const chess::Move& move) {
  int from = move.from().index();
  int to = move.to().index();
  
  int from_rank = from / 8;
  int from_file = from % 8;  
  int to_rank = to / 8;
  int to_file = to % 8;
  
  // Map to 64×64 grid coordinates  
  int x = from_rank * 8 + to_rank;
  int y = from_file * 8 + to_file;
  
  return x * 64 + y;
}
```

### 2. Update Model Architecture (`model.py`)
Replace linear policy head with convolutional layers:
```python
# Current: nn.Linear(32 * 8 * 8, 64 * 64)
# New: Spatial conv layers
self.policy_head = nn.Sequential(
    nn.Conv2d(128, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(), 
    nn.Conv2d(64, 64, kernel_size=1),
    # Reshape to maintain spatial structure
)
```

## Benefits
- **Spatial awareness**: Conv layers can learn move patterns and tactical relationships
- **Parameter efficiency**: Fewer parameters than large fully-connected layer  
- **Pattern recognition**: Direct learning of tactical motifs in move space
- **Natural structure**: 64×64 grid represents inherent spatial move relationships

## Required Changes
- [ ] `chess_utils.hpp`: Update `move_to_int()` and `int_to_move()` functions
- [ ] `model.py`: Replace linear policy head with conv layers
- [ ] `test/mate_in_one.py`: Update to use new encoding scheme
- [ ] All C++ code using move encoding (worker.cpp, evaluator.cpp, etc.)
- [ ] Training pipeline: Handle new move encoding format

## Critical Considerations
- **Breaking change**: All existing training data becomes invalid
- **Model compatibility**: All existing models become incompatible
- **System-wide impact**: Move encoding affects MCTS, training, evaluation
- **Testing required**: Extensive validation of new encoding correctness

## Evaluation Plan
**REQUIRED**: Before/after model performance comparison to validate improvement:

1. **Baseline**: Train current architecture for N iterations, measure:
   - Training loss convergence
   - Test position accuracy (mate-in-one, tactics)
   - Self-play ELO progression

2. **New architecture**: Train spatial encoding architecture for same N iterations
3. **Head-to-head**: Direct ELO comparison between architectures
4. **Analysis**: Compare learned patterns, training efficiency, final strength

## Priority
Medium - Architectural improvement requiring careful evaluation

## Status
- [ ] Not started
- [ ] Research phase
- [ ] Implementation
- [ ] Testing
- [ ] Evaluation
- [ ] Complete

## Notes
This is a fundamental change to move representation. Requires careful implementation and thorough evaluation to ensure the spatial approach actually improves model performance before adopting.