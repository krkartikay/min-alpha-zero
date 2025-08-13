#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/log/log_sink_registry.h>

#include "alpha_zero.h"

ABSL_FLAG(int, channel_size, 128, "Channel size (must be power of 2)");
ABSL_FLAG(int, num_simulations, 200, "Number of MCTS simulations per move");
ABSL_FLAG(int, batch_size, 1000, "Number of nodes to process at once");
ABSL_FLAG(int, eval_timeout_ms, 1,
          "Timeout for evaluation requests in milliseconds");
ABSL_FLAG(int, num_games, 1, "Number of games to play in self-play");
ABSL_FLAG(int, num_threads, 1, "Number of worker threads to start");
ABSL_FLAG(std::string, model_path, "model.pt", "Path to the model file");
ABSL_FLAG(std::string, training_file, "training_data.bin",
          "File to store training data");
ABSL_FLAG(bool, debug, false,
          "Enable debug mode to dump game tree to file after each move");

namespace alphazero {

void init_globals() {
  g_config.channel_size = absl::GetFlag(FLAGS_channel_size);
  g_config.num_simulations = absl::GetFlag(FLAGS_num_simulations);
  g_config.batch_size = absl::GetFlag(FLAGS_batch_size);
  g_config.eval_timeout =
      duration<int, milli>(absl::GetFlag(FLAGS_eval_timeout_ms));
  g_config.num_games = absl::GetFlag(FLAGS_num_games);
  g_config.num_threads = absl::GetFlag(FLAGS_num_threads);
  g_config.model_path = absl::GetFlag(FLAGS_model_path);
  g_config.training_file = absl::GetFlag(FLAGS_training_file);
  g_config.debug = absl::GetFlag(FLAGS_debug);
  // initialize the evaluation queue with the channel size
  g_evaluation_queue = std::make_unique<eval_channel_t>(g_config.channel_size);
}

}  // namespace alphazero

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  alphazero::FileSink file_sink("alpha_zero.log");
  absl::AddLogSink(&file_sink);

  alphazero::init_globals();

  // Create boost channel
  LOG(INFO) << "Starting AlphaZero...";

  // Start evaluator thread
  LOG(INFO) << "Starting Evaluator thread.";
  std::thread evaluator_thread(&alphazero::run_evaluator);

  // Start multiple worker threads based on num_threads parameter
  LOG(INFO) << absl::StrFormat("Starting %d worker threads.", 
                               alphazero::g_config.num_threads);
  std::vector<std::thread> worker_threads;
  worker_threads.reserve(alphazero::g_config.num_threads);
  
  for (int i = 0; i < alphazero::g_config.num_threads; ++i) {
    worker_threads.emplace_back(&alphazero::run_worker);
  }

  // Wait for all worker threads to complete
  for (auto& worker_thread : worker_threads) {
    worker_thread.join();
  }

  alphazero::g_stop_evaluator = true;  // Signal evaluator to stop
  evaluator_thread.join();

  return 0;
}