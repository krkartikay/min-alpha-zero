#include <torch/script.h>
#include <torch/torch.h>

#include "alpha_zero.h"

inline torch::jit::script::Module g_model;

namespace alphazero {

// Evaluator thread main loop
void run_evaluator() {
  // Evaluator is supposed to get a batch of requests at once
  // and then evaluate them and send the results back.
  LOG(INFO) << "Loading model...";
  init_model();
  LOG(INFO) << "Evaluator started.";
  while (!g_stop_evaluator) {
    std::vector<eval_request_t> req_batch = get_requests_batch();
    process_batch(std::move(req_batch));
  }
}

void init_model() {
  g_model = torch::jit::load(g_config.model_path);
  g_model.eval();
  g_model.to(torch::kCUDA);
}

std::vector<eval_request_t> get_requests_batch() {
  // gets a batch of requests from global queue with timeout
  std::vector<eval_request_t> req_batch;
  eval_request_t req;
  time_point_t timeout = steady_clock::now() + g_config.eval_timeout;
  while (int(req_batch.size()) < g_config.batch_size) {
    channel_op_status status = g_evaluation_queue->pop_wait_until(req, timeout);
    if (status == channel_op_status::success) {
      req_batch.push_back(std::move(req));
    } else if (status == channel_op_status::timeout) {
      break;  // Timeout reached, exit the loop
    }
  }
  return req_batch;
}

void process_batch(std::vector<eval_request_t> nodes) {
  // Copy Tensors into a single batch tensor
  const size_t batch_size = nodes.size();
  torch::Tensor input = torch::empty({int(batch_size), 7, 8, 8},
                                     torch::TensorOptions().dtype(torch::kF32));

  // Convert boards to tensors and copy
  for (size_t i = 0; i < batch_size; ++i) {
    std::array<float, kInputSize> buf = board_to_tensor(nodes[i].first->board);
    std::memcpy(input[i].data_ptr(), buf.data(), kInputSize * sizeof(float));
  }

  // Move input to CUDA and run the model
  auto input_cuda = input.to(torch::kCUDA);
  auto output = g_model.forward({input_cuda});

  // The model output is a tuple of (policy, value) batched tensors
  auto output_tuple = output.toTuple();
  auto policy_batch = output_tuple->elements()[0].toTensor().to(torch::kCPU);
  auto value_batch = output_tuple->elements()[1].toTensor().to(torch::kCPU);

  // Copy results back to nodes and signal completion
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto& [node, p] = nodes[i];
    node->value = value_batch[i].item<float>();
    std::memcpy(node->policy.data(), policy_batch[i].data_ptr(),
                kNumActions * sizeof(float));
    node->is_evaluated = true;
    p.set_value();
  }
}

}  // namespace alphazero