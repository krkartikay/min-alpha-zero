#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/torch.h>

#include <boost/fiber/all.hpp>
#include <iostream>

#ifndef ALPHA_ZERO_H
#define ALPHA_ZERO_H

namespace alphazero {

using boost::fibers::buffered_channel;
using boost::fibers::channel_op_status;
using boost::fibers::future;
using boost::fibers::promise;

using eval_request_t = std::pair<int, boost::fibers::promise<int>>;
using eval_channel_t = boost::fibers::buffered_channel<eval_request_t>;

class Evaluator {
 public:
  Evaluator(eval_channel_t& channel) : channel_(channel) {}
  void run();

 private:
  eval_channel_t& channel_;
};

class Worker {
 public:
  Worker(eval_channel_t& channel) : channel_(channel) {}
  void run();
  void work_on(int i);

  future<int> send_request(int request);

 private:
  eval_channel_t& channel_;
};

}  // namespace alphazero

#endif  // ALPHA_ZERO_H