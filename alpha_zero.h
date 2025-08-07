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

// Global Position Evaluation queue --------------------------

constexpr int kChannelSize = 128;  // Must be a power of 2!
inline eval_channel_t g_evaluation_queue(kChannelSize);

// Evaluator thread ------------------------------------------

void run_evaluator();

// Worker thread ---------------------------------------------

void run_worker();
future<int> send_request(int request);
void work_on(int i);

// -----------------------------------------------------------

}  // namespace alphazero

#endif  // ALPHA_ZERO_H