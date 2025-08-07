#include "alpha_zero.h"

namespace alphazero {

// Worker thread main loop
void Worker::run() {
  // Starts a number of fibers, each one sends as many requests
  // to the channel as possible.
  for (int i = 0; i < 100; ++i) {
    boost::fibers::fiber([this, i]() { work_on(i); }).detach();
  }
}

void Worker::work_on(int i) {
  // Send a request to the channel
  future<int> response = send_request(i);
  // Wait for the response
  int result = response.get();
  // Print the result
  std::cout << "Worker sent: " << i << " and got: " << result << std::endl;
}

future<int> Worker::send_request(int request) {
  // Create a promise and future pair
  promise<int> promise;
  future<int> future = promise.get_future();

  // Send the request to the channel
  channel_.push(std::make_pair(request, std::move(promise)));
  return future;
}

}  // namespace alphazero