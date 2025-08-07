#include "alpha_zero.h"

const int kChannelSize = 128;  // Must be a power of 2!

int main() {
  // Create boost channel
  std::cout << "Starting AlphaZero..." << std::endl;
  std::cout << "Creating channel with size: " << kChannelSize << std::endl;
  alphazero::eval_channel_t channel(kChannelSize);

  // Create evaluator and worker instances
  std::cout << "Creating Evaluator and worker." << std::endl;
  alphazero::Evaluator evaluator(channel);
  alphazero::Worker worker(channel);

  // Start evaluator and worker threads on actual system threads
  std::cout << "Starting Evaluator and worker threads." << std::endl;
  std::thread evaluator_thread(&alphazero::Evaluator::run, &evaluator);
  std::thread worker_thread(&alphazero::Worker::run, &worker);

  evaluator_thread.join();
  worker_thread.join();

  return 0;
}