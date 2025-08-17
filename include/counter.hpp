#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace metrics {

class Counter {
 public:
  static Counter& GetInstance() {
    static Counter instance;
    return instance;
  }

  void Increment(const std::string& counter_name, int value = 1) {
    std::lock_guard<std::mutex> lock(counters_mutex_);
    auto it = counters_.find(counter_name);
    if (it == counters_.end()) {
      counters_[counter_name] = std::make_shared<std::atomic<int>>(0);
      it = counters_.find(counter_name);
    }
    it->second->fetch_add(value);
  }

  void Start() {
    if (!running_.exchange(true)) {
      logger_thread_ = std::make_unique<std::thread>(&Counter::BackgroundLogger, this);
    }
  }

  void Stop() {
    if (running_.exchange(false)) {
      if (logger_thread_ && logger_thread_->joinable()) {
        logger_thread_->join();
      }
    }
  }

  ~Counter() {
    Stop();
  }

 private:
  Counter() {
    Start();
  }

  Counter(const Counter&) = delete;
  Counter& operator=(const Counter&) = delete;

  void BackgroundLogger() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      if (!running_) break;

      std::lock_guard<std::mutex> lock(counters_mutex_);

      std::vector<std::string> metric_strings;
      for (const auto& [name, counter] : counters_) {
        int value = counter->exchange(0);
        if (value > 0) {
          metric_strings.push_back(absl::StrCat(name, "=", value));
        }
      }

      if (!metric_strings.empty()) {
        LOG(INFO) << "[METRICS] " << absl::StrJoin(metric_strings, " ");
      }
    }
  }

  std::unordered_map<std::string, std::shared_ptr<std::atomic<int>>> counters_;
  std::mutex counters_mutex_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> logger_thread_;
};

inline void Increment(const std::string& counter_name, int value = 1) {
  Counter::GetInstance().Increment(counter_name, value);
}

}  // namespace metrics