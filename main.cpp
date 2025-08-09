#include "alpha_zero.h"

int main() {
  // Create boost channel
  std::cout << "Starting AlphaZero..." << std::endl;

  // Start evaluator and worker threads on actual system threads
  std::cout << "Starting Evaluator and worker threads." << std::endl;
  std::thread evaluator_thread(&alphazero::run_evaluator);
  std::thread worker_thread(&alphazero::run_worker);

  worker_thread.join();

  alphazero::g_stop_evaluator = true;  // Signal evaluator to stop
  evaluator_thread.join();

  return 0;
}