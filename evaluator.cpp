#include <torch/script.h>
#include <torch/torch.h>

#include "alpha_zero.h"

namespace alphazero {

torch::jit::script::Module g_model;

struct PreparedBatch {
  std::vector<eval_request_t> reqs;       // Node* + promise
  torch::Tensor               cpu_tensor; // [B,7,8,8] on host
};

struct InferredBatch {
  std::vector<eval_request_t> reqs;   // Node* + promise
  torch::Tensor policy_cuda;          // [B,kNumActions]  (soft-maxed, still on GPU)
  torch::Tensor value_cuda;           // [B]              (on GPU)
};

using batch_channel_t = boost::fibers::buffered_channel<PreparedBatch>;
using out_channel_t = boost::fibers::buffered_channel<InferredBatch>;

std::unique_ptr<batch_channel_t> g_gpu_queue;
std::unique_ptr<out_channel_t> g_out_queue;

// Evaluator thread main loop
void run_evaluator() {
  // Evaluator is supposed to get a batch of requests at once
  // and then evaluate them and send the results back.
  LOG(INFO) << "Loading model...";
  init_model();
  LOG(INFO) << "Evaluator started.";

  std::thread gpu_thr(gpu_stage);
  std::thread post_thr(post_stage);
  cpu_stage();  // current thread
  gpu_thr.join();
  post_thr.join();
}

void init_model() {
  g_model = torch::jit::load(g_config.model_path);
  g_model.eval();
  g_model.to(torch::kCUDA);
  g_gpu_queue = std::make_unique<batch_channel_t>(g_config.channel_size);
  g_out_queue = std::make_unique<out_channel_t >(g_config.channel_size);
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

void cpu_stage() {
  while (!g_stop_evaluator) {
    auto batch_reqs = get_requests_batch();
    if (batch_reqs.empty()) continue;

    int B = static_cast<int>(batch_reqs.size());
    torch::Tensor cpu = torch::empty({B, 7, 8, 8},
                         torch::dtype(torch::kF32));

    for (int i = 0; i < B; ++i) {
      auto buf = board_to_tensor(batch_reqs[i].first->board);
      std::memcpy(cpu[i].data_ptr(), buf.data(),
                  kInputSize * sizeof(float));
    }
    g_gpu_queue->push({std::move(batch_reqs), std::move(cpu)});
  }
  g_gpu_queue->close();
}

void gpu_stage() {
  PreparedBatch pb;
  while (g_gpu_queue->pop(pb) == channel_op_status::success) {
    auto output = g_model.forward({pb.cpu_tensor.to(torch::kCUDA)});
    auto tup    = output.toTuple();

    auto policy_cuda = torch::softmax(tup->elements()[0].toTensor(), 1);
    auto value_cuda  = tup->elements()[1].toTensor();

    g_out_queue->push({std::move(pb.reqs),
                       std::move(policy_cuda),
                       std::move(value_cuda)});
  }
  g_out_queue->close();
}

void post_stage() {
  InferredBatch ib;
  while (g_out_queue->pop(ib) == channel_op_status::success) {
    // async copy to cpu
    auto policy_cpu = ib.policy_cuda.to(torch::kCPU, true);
    auto value_cpu  = ib.value_cuda .to(torch::kCPU, true);
    torch::cuda::synchronize();

    for (size_t i = 0; i < ib.reqs.size(); ++i) {
      auto& [node, prom] = ib.reqs[i];
      node->value = value_cpu[i].item<float>();
      std::memcpy(node->policy.data(),
                  policy_cpu[i].data_ptr(),
                  kNumActions * sizeof(float));
      node->is_evaluated = true;
      prom.set_value();
    }
  }
}

}  // namespace alphazero