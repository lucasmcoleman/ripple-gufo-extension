#include "hip/gpu_executor_factory.hpp"

#include <hip/hip_runtime.h>

#include <exception>
#include <iterator>
#include <map>
#include <mutex>
#include <utility>

#include "hip/device_model.hpp"
#include "hip/executor.hpp"
#include "src/executor_factory.hpp"

#if !defined(RICURSIVE_HAVE_GPU_EXECUTOR)
#error "gpu_executor_factory.cpp must be built with RICURSIVE_HAVE_GPU_EXECUTOR (extension.cmake)"
#endif

namespace gufo::models::ricursive {
namespace {

/// What every executor of one model keeps alive: the mapped GGUF (the host
/// embedding table and the upload sources point into it) and the device copy.
struct SharedModel {
  std::shared_ptr<const core::GgufReader> reader;
  std::unique_ptr<DeviceModel> device;
};

struct Slot {
  std::weak_ptr<const core::GgufReader> reader;  ///< registered by the engine
  std::weak_ptr<SharedModel> model;              ///< the uploaded copy, if alive
};

std::mutex g_mutex;

/// Never destroyed: executors may outlive static destruction in a test or
/// tool, and the slots hold weak references only.
std::map<const ModelWeights*, Slot>& Slots() {
  static auto* slots = new std::map<const ModelWeights*, Slot>();
  return *slots;
}

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

/// The shared model for `weights`: alive copy, else upload now. Under g_mutex
/// so two sessions starting together upload once.
std::shared_ptr<SharedModel> Acquire(const ModelWeights& weights, std::string* error) {
  const std::lock_guard lock(g_mutex);
  auto it = Slots().find(&weights);
  if (it == Slots().end()) {
    SetError(error,
             "GPU executor: the weights were not registered with "
             "RegisterGpuWeightsReader (the factory needs the GGUF reader they came from)");
    return nullptr;
  }
  Slot& slot = it->second;
  if (auto alive = slot.model.lock()) return alive;
  auto reader = slot.reader.lock();
  if (!reader) {
    SetError(error, "GPU executor: the GGUF reader of these weights is gone");
    return nullptr;
  }
  std::string upload_error;
  auto device = DeviceModel::Upload(weights, *reader, &upload_error);
  if (!device) {
    SetError(error, "GPU executor: weight upload failed: " + upload_error);
    return nullptr;
  }
  auto shared = std::make_shared<SharedModel>();
  shared->reader = std::move(reader);
  shared->device = std::move(device);
  slot.model = shared;
  return shared;
}

}  // namespace

void RegisterGpuWeightsReader(const ModelWeights& weights,
                              std::shared_ptr<const core::GgufReader> reader) {
  const std::lock_guard lock(g_mutex);
  auto& slots = Slots();
  for (auto it = slots.begin(); it != slots.end();) {  // forget dead models
    it = it->second.reader.expired() && it->second.model.expired() ? slots.erase(it) : std::next(it);
  }
  Slot& slot = slots[&weights];
  if (slot.reader.lock() == reader && reader != nullptr) return;
  slot.reader = reader;
  slot.model.reset();  // another file behind the same address: never reuse its upload
}

std::size_t GpuSharedModelCount() {
  const std::lock_guard lock(g_mutex);
  std::size_t n = 0;
  for (const auto& [key, slot] : Slots()) n += slot.model.expired() ? 0 : 1;
  return n;
}

bool GpuMemoryUsage(std::size_t* used_bytes, std::size_t* total_bytes, std::string* error) {
  std::size_t free_bytes = 0, total = 0;
  const hipError_t status = hipMemGetInfo(&free_bytes, &total);
  if (status != hipSuccess) {
    SetError(error, std::string("hipMemGetInfo: ") + hipGetErrorString(status));
    return false;
  }
  if (used_bytes != nullptr) *used_bytes = total - free_bytes;
  if (total_bytes != nullptr) *total_bytes = total;
  return true;
}

std::shared_ptr<const DeviceModel> AcquireGpuTargetModel(const ModelWeights& weights, std::string* error) {
  try {
    std::shared_ptr<SharedModel> shared = Acquire(weights, error);
    if (!shared) return nullptr;
    return std::shared_ptr<const DeviceModel>(shared, shared->device.get());
  } catch (const std::exception& e) {
    SetError(error, std::string("GPU executor: ") + e.what());
  }
  return nullptr;
}

std::unique_ptr<Executor> CreateGpuExecutor(const ModelWeights& weights,
                                            const ExecutorConfig& config,
                                            std::string* error) {
  try {
    // Refuse what the executor cannot run (head_dim, GQA group, clamp, ...)
    // before anything is uploaded; max_context 0 becomes min(32768, GGUF).
    // kv_type is not touched: F16 unless the caller asks for F32 (test mode).
    const ExecutorConfig normalized = NormalizeConfig(weights.config, config);
    std::shared_ptr<SharedModel> shared = Acquire(weights, error);
    if (!shared) return nullptr;
    // Aliasing constructor: the executor's keep-alive owns the whole SharedModel.
    std::shared_ptr<const DeviceModel> device(shared, shared->device.get());
    return std::make_unique<GpuExecutor>(std::move(device), normalized);
  } catch (const std::exception& e) {
    SetError(error, std::string("GPU executor: ") + e.what());
  } catch (...) {
    SetError(error, "GPU executor: unknown failure");
  }
  return nullptr;
}

}  // namespace gufo::models::ricursive
