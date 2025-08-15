#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <alpha_zero.h>
#include <chess_utils.hpp>

namespace py = pybind11;

PYBIND11_MODULE(min_alpha_zero, m) {
    m.doc() = "Python bindings for min-alpha-zero chess engine";

    // Minimal chess bindings needed for alpha_zero types
    py::class_<chess::Board>(m, "Board")
        .def(py::init<>())
        .def("sideToMove", &chess::Board::sideToMove);
        
    py::class_<chess::Move>(m, "Move");
    
    py::class_<chess::Color>(m, "Color")
        .def(py::init<>())
        .def_readonly_static("WHITE", &chess::Color::WHITE)
        .def_readonly_static("BLACK", &chess::Color::BLACK);

    // Constants
    m.attr("NUM_ACTIONS") = alphazero::kNumActions;
    m.attr("INPUT_SIZE") = alphazero::kInputSize;

    // Config class
    py::class_<alphazero::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("channel_size", &alphazero::Config::channel_size)
        .def_readwrite("num_simulations", &alphazero::Config::num_simulations)
        .def_readwrite("batch_size", &alphazero::Config::batch_size)
        .def_readwrite("num_games", &alphazero::Config::num_games)
        .def_readwrite("num_threads", &alphazero::Config::num_threads)
        .def_readwrite("model_path", &alphazero::Config::model_path)
        .def_readwrite("training_file", &alphazero::Config::training_file)
        .def_readwrite("debug", &alphazero::Config::debug);

    // Node class
    py::class_<alphazero::Node>(m, "Node")
        .def(py::init<const chess::Board&>(), py::arg("board") = chess::Board())
        .def_readonly("board", &alphazero::Node::board)
        .def_readonly("is_evaluated", &alphazero::Node::is_evaluated)
        .def_readonly("is_leaf", &alphazero::Node::is_leaf)
        .def_readonly("parent_action", &alphazero::Node::parent_action)
        .def_readonly("value", &alphazero::Node::value)
        .def_readonly("move_history", &alphazero::Node::move_history)
        .def("select_action", &alphazero::Node::selectAction)
        .def("evaluate", &alphazero::Node::evaluate)
        .def("evaluate_leaf_node", &alphazero::Node::evaluateLeafNode)
        .def("get_child_node", &alphazero::Node::getChildNode, py::return_value_policy::reference);

    // GameState class
    py::class_<alphazero::GameState>(m, "GameState")
        .def(py::init<>())
        .def_readonly("value", &alphazero::GameState::value)
        .def_readonly("final_value", &alphazero::GameState::final_value);

    // Game class
    py::class_<alphazero::Game>(m, "Game")
        .def(py::init<>())
        .def("self_play", &alphazero::Game::selfPlay)
        .def("update_root", &alphazero::Game::updateRoot)
        .def("save_game_state", &alphazero::Game::saveGameState)
        .def("update_game_history", &alphazero::Game::updateGameHistory)
        .def("select_move", &alphazero::Game::selectMove)
        .def("run_simulation", &alphazero::Game::runSimulation)
        .def("append_to_training_file", &alphazero::Game::appendToTrainingFile);

    // ChessAgent base class
    py::class_<alphazero::ChessAgent>(m, "ChessAgent")
        .def("select_action", &alphazero::ChessAgent::select_action)
        .def("name", &alphazero::ChessAgent::name);

    // RandomAgent class
    py::class_<alphazero::RandomAgent, alphazero::ChessAgent>(m, "RandomAgent")
        .def(py::init<>())
        .def("select_action", &alphazero::RandomAgent::select_action)
        .def("name", &alphazero::RandomAgent::name);

    // MCTSAgent class
    py::class_<alphazero::MCTSAgent, alphazero::ChessAgent>(m, "MCTSAgent")
        .def(py::init<>())
        .def("select_action", &alphazero::MCTSAgent::select_action)
        .def("name", &alphazero::MCTSAgent::name);

    // GameResult class
    py::class_<alphazero::GameResult>(m, "GameResult")
        .def(py::init<>())
        .def_readonly("moves_played", &alphazero::GameResult::moves_played)
        .def_readonly("agent_wins", &alphazero::GameResult::agent_wins)
        .def_readonly("draws", &alphazero::GameResult::draws)
        .def_readonly("other_wins", &alphazero::GameResult::other_wins);

    // Free functions from alpha_zero.h
    m.def("run_evaluator", &alphazero::run_evaluator);
    m.def("init_model", &alphazero::init_model);
    m.def("run_worker", &alphazero::run_worker);
    m.def("play_agent_vs_agent", &alphazero::play_agent_vs_agent);
    m.def("run_agent_tournament", &alphazero::run_agent_tournament);
    m.def("dump_game_tree_to_file", &alphazero::dump_game_tree_to_file,
          py::arg("game"), py::arg("g") = 0, py::arg("m") = 0, py::arg("chosen_action") = -1);
    m.def("board_to_string", &alphazero::board_to_string);
    m.def("timestamp", &alphazero::timestamp);

    // Functions from chess_utils.hpp
    m.def("board_to_tensor", &chess::board_to_tensor);
    m.def("move_to_int", &chess::move_to_int);
    m.def("int_to_move", &chess::int_to_move);

    // Global config access
    m.def("get_config", []() -> alphazero::Config& {
        return alphazero::g_config;
    }, py::return_value_policy::reference);
}