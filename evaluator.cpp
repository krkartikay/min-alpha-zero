#include "alpha_zero.h"

namespace alphazero {

// Evaluator thread main loop
void run_evaluator() {
  // Evaluator is supposed to get a batch of requests at once
  // and then evaluate them and send the results back.
  std::cout << "Evaluator started." << std::endl;
  while (true) {
    eval_request_t req;
    channel_op_status status = g_evaluation_queue.pop(req);
    if (status == channel_op_status::success) {
      int input = req.first;
      // let's wait a bit to simulate some processing time
      absl::SleepFor(absl::Milliseconds(100));
      req.second.set_value(input * 2);  // Return dummy value
    }
  }
}

}  // namespace alphazero