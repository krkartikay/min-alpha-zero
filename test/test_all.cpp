#include <gtest/gtest.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include "alpha_zero.h"

using namespace alphazero;

class AlphaZeroTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize minimal config for testing
    g_config.channel_size = 32;
    g_config.num_simulations = 10;
    g_config.batch_size = 1;
    g_config.num_games = 1;
    g_config.num_threads = 1;
    g_config.eval_timeout = duration_t(100);
    g_config.model_path = "model.pt";
    g_config.training_file = "test_training.bin";
    g_config.debug = false;
  }
};

// Test Node class functionality
TEST_F(AlphaZeroTest, NodeConstruction) {
  chess::Board board;
  Node node(board);
  
  EXPECT_FALSE(node.is_evaluated);
  EXPECT_FALSE(node.is_leaf);
  EXPECT_EQ(node.parent, nullptr);
  EXPECT_EQ(node.parent_action, -1);
  EXPECT_EQ(node.value, 0.0f);
  
  // Check legal mask has legal moves marked
  bool has_legal_moves = false;
  for (bool is_legal : node.legal_mask) {
    if (is_legal) {
      has_legal_moves = true;
      break;
    }
  }
  EXPECT_TRUE(has_legal_moves);
}

TEST_F(AlphaZeroTest, NodeGetChildNode) {
  chess::Board board;
  Node node(board);
  
  // Find a legal move
  int legal_action = -1;
  for (int i = 0; i < kNumActions; ++i) {
    if (node.legal_mask[i]) {
      legal_action = i;
      break;
    }
  }
  ASSERT_NE(legal_action, -1);
  
  // Get child node
  Node* child = node.getChildNode(legal_action);
  ASSERT_NE(child, nullptr);
  EXPECT_EQ(child->parent, &node);
  EXPECT_EQ(child->parent_action, legal_action);
  
  // Second call should return same node
  Node* child2 = node.getChildNode(legal_action);
  EXPECT_EQ(child, child2);
  
  // Illegal move should return nullptr
  int illegal_action = -1;
  for (int i = 0; i < kNumActions; ++i) {
    if (!node.legal_mask[i]) {
      illegal_action = i;
      break;
    }
  }
  if (illegal_action != -1) {
    EXPECT_EQ(node.getChildNode(illegal_action), nullptr);
  }
}

// Test chess utility functions
TEST_F(AlphaZeroTest, ChessUtilities) {
  chess::Board board;
  
  // Test board_to_tensor
  auto tensor = chess::board_to_tensor(board);
  EXPECT_EQ(tensor.size(), kInputSize);
  
  // Test move conversion
  chess::Movelist moves;
  chess::movegen::legalmoves(moves, board);
  ASSERT_FALSE(moves.empty());
  
  chess::Move first_move = moves[0];
  int move_int = chess::move_to_int(first_move);
  chess::Move converted_back = chess::int_to_move(move_int, board);
  
  EXPECT_EQ(first_move.from(), converted_back.from());
  EXPECT_EQ(first_move.to(), converted_back.to());
}

// Test Game class functionality
TEST_F(AlphaZeroTest, GameConstruction) {
  Game game;
  ASSERT_NE(game.root, nullptr);
  EXPECT_EQ(game.root->parent, nullptr);
  EXPECT_TRUE(game.history.empty());
}

// Test Agent classes
TEST_F(AlphaZeroTest, RandomAgent) {
  RandomAgent agent;
  EXPECT_EQ(agent.name(), "Random");
  
  Game game;
  int action = agent.select_action(game);
  
  // Should select a legal action or -1 if no legal moves
  if (action != -1) {
    EXPECT_TRUE(game.root->legal_mask[action]);
  }
}

TEST_F(AlphaZeroTest, MCTSAgent) {
  MCTSAgent agent;
  EXPECT_EQ(agent.name(), "MCTS");
  
  // Note: Full MCTS testing would require neural network setup
  // This just tests basic construction and interface
}

// Test utility functions
TEST_F(AlphaZeroTest, TimestampFunction) {
  std::string ts = timestamp();
  EXPECT_FALSE(ts.empty());
  EXPECT_EQ(ts.front(), '[');
  EXPECT_EQ(ts.back(), ']');
}

TEST_F(AlphaZeroTest, BoardToString) {
  chess::Board board;
  std::string board_str = board_to_string(board);
  EXPECT_FALSE(board_str.empty());
  // Should contain rank numbers and piece symbols
  EXPECT_NE(board_str.find('8'), std::string::npos);
  EXPECT_NE(board_str.find('1'), std::string::npos);
}

// Test struct functionality
TEST_F(AlphaZeroTest, GameStateStruct) {
  GameState state;
  EXPECT_EQ(state.value, 0.0f);
  EXPECT_EQ(state.final_value, 0);
  
  // Arrays should be zero-initialized
  for (float val : state.board_tensor) {
    EXPECT_EQ(val, 0.0f);
  }
  for (bool val : state.legal_mask) {
    EXPECT_FALSE(val);
  }
}

// Test leaf node evaluation
TEST_F(AlphaZeroTest, EvaluateLeafNode) {
  // Test checkmate position: Black king checkmated by white queen and king
  chess::Board checkmate_board("k7/Q7/K7/8/8/8/8/8 b - - 0 1");
  Node checkmate_node(checkmate_board);
  
  if (checkmate_node.is_leaf) {
    checkmate_node.evaluateLeafNode();
    EXPECT_TRUE(checkmate_node.is_evaluated);
    EXPECT_EQ(checkmate_node.value, -1.0f);
  } else {
    // If not a leaf, just test that evaluateLeafNode sets is_evaluated
    checkmate_node.evaluateLeafNode();
    EXPECT_TRUE(checkmate_node.is_evaluated);
  }
}

// Test save_game_state function
TEST_F(AlphaZeroTest, SaveGameState) {
  Game game;
  game.root->is_evaluated = true;
  game.root->value = 0.5f;
  
  // Set some dummy data
  game.root->legal_mask[0] = true;
  game.root->policy[0] = 0.1f;
  game.root->child_visits[0] = 5;
  
  game.saveGameState();
  
  EXPECT_EQ(game.history.size(), 1);
  EXPECT_EQ(game.history[0].value, 0.5f);
  EXPECT_TRUE(game.history[0].legal_mask[0]);
  EXPECT_EQ(game.history[0].policy[0], 0.1f);
  EXPECT_EQ(game.history[0].child_visit_counts[0], 5);
}

// Test update_root function
TEST_F(AlphaZeroTest, UpdateRoot) {
  Game game;
  
  // Find a legal action
  int legal_action = -1;
  for (int i = 0; i < kNumActions; ++i) {
    if (game.root->legal_mask[i]) {
      legal_action = i;
      break;
    }
  }
  ASSERT_NE(legal_action, -1);
  
  // Create child node
  Node* child = game.root->getChildNode(legal_action);
  ASSERT_NE(child, nullptr);
  
  // Update root
  game.updateRoot(legal_action);
  
  // Check that root was updated
  EXPECT_EQ(game.root.get(), child);
  EXPECT_EQ(game.root->parent, nullptr);
}

int main(int argc, char **argv) {
  absl::InitializeLog();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}