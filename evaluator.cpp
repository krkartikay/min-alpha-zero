#include "alpha_zero.h"

namespace alphazero {

// Evaluator thread main loop
void run_evaluator() {
  // Evaluator is supposed to get a batch of requests at once
  // and then evaluate them and send the results back.
  std::cout << "Evaluator started." << std::endl;
  while (true) {
    std::vector<eval_request_t> req_batch = get_requests_batch();
    process_batch(std::move(req_batch));
  }
}

std::vector<eval_request_t> get_requests_batch() {
  // gets a batch of requests from global queue with timeout
  std::vector<eval_request_t> req_batch;
  eval_request_t req;
  time_point_t timeout = steady_clock::now() + kEvalTimeout;
  while (req_batch.size() < kBatchSize) {
    channel_op_status status = g_evaluation_queue.pop_wait_until(req, timeout);
    if (status == channel_op_status::success) {
      req_batch.push_back(std::move(req));
    } else if (status == channel_op_status::timeout) {
      break;  // Timeout reached, exit the loop
    }
  }
  return req_batch;
}

void process_batch(std::vector<eval_request_t> nodes) {
  std::cout << absl::StrFormat("Processing batch of %d nodes", nodes.size())
            << std::endl;
  absl::SleepFor(absl::Milliseconds(1));
  for (auto& [node, p] : nodes) {
    node->is_evaluated = true;
    p.set_value();
  }
}

}  // namespace alphazero