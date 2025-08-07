#include "alpha_zero.h"

namespace alphazero {

// Evaluator thread main loop
void Evaluator::run() {
  // Evaluator is supposed to get a batch of requests at once
  // and then evaluate them and send the results back.
  std::cout << "Evaluator started." << std::endl;
  while (true) {
    std::cout << "Evaluator trying to fetch request..." << std::endl;
    eval_request_t req;
    channel_op_status status = channel_.pop(req);
    std::cout << "Evaluator fetched request." << std::endl;
    if (status == channel_op_status::success) {
      std::cout << "Evaluator req success, processing " << req.first
                << std::endl;
      int input = req.first;
      req.second.set_value(input * 2);  // Return dummy value
    } else {
      std::cout << "Evaluator req code not success: " << int(status)
                << std::endl;
    }
  }
}

}  // namespace alphazero