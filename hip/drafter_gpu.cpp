#include "hip/drafter_gpu.hpp"

#include <exception>
#include <map>
#include <mutex>

#include "hip/device_model.hpp"
#include "hip/gpu_executor_factory.hpp"

namespace gufo::models::ricursive {
namespace {

/// What every engine of one drafter keeps alive: the mapped GGUF (the host views and the upload
/// sources point into it) and the device copy.
struct SharedDrafter {
  std::shared_ptr<const core::GgufReader> reader;
  std::unique_ptr<DeviceDrafter> device;
};

struct Slot {
  std::weak_ptr<const core::GgufReader> reader;
  std::weak_ptr<SharedDrafter> model;
};

std::mutex g_mutex;
/// Never destroyed (engines may outlive static destruction); weak references only. Keyed by the
/// address of the weights table AND checked against the reader, so another file behind a reused
/// address never gets the previous upload (docs LESSONS: device-model cache keyed by an address).
std::map<const DrafterWeights*, Slot>& Slots() {
  static auto* slots = new std::map<const DrafterWeights*, Slot>();
  return *slots;
}

std::shared_ptr<SharedDrafter> Acquire(const DrafterWeights& weights,
                                       const std::shared_ptr<const core::GgufReader>& reader, std::string* error) {
  const std::lock_guard lock(g_mutex);
  auto& slots = Slots();
  for (auto it = slots.begin(); it != slots.end();) {  // forget dead drafters
    it = it->second.reader.expired() && it->second.model.expired() ? slots.erase(it) : std::next(it);
  }
  Slot& slot = slots[&weights];
  if (slot.reader.lock() != reader) {
    slot.reader = reader;
    slot.model.reset();
  }
  if (auto alive = slot.model.lock()) return alive;
  auto device = DeviceDrafter::Upload(weights, *reader, error);
  if (!device) return nullptr;
  auto shared = std::make_shared<SharedDrafter>();
  shared->reader = reader;
  shared->device = std::move(device);
  slot.model = shared;
  return shared;
}

struct KeepAlive {
  std::shared_ptr<SharedDrafter> drafter;
  std::shared_ptr<const DeviceModel> target;
  std::shared_ptr<const core::GgufReader> target_reader;
};

std::unique_ptr<DrafterEngine> Build(const std::shared_ptr<SharedDrafter>& drafter,
                                     std::shared_ptr<const DeviceModel> target,
                                     std::shared_ptr<const core::GgufReader> target_reader,
                                     const DrafterOptions& options, const DrafterExecFaults& faults) {
  auto keep = std::make_shared<KeepAlive>();
  keep->drafter = drafter;
  keep->target = std::move(target);
  keep->target_reader = std::move(target_reader);
  const ModelHandles& head = keep->target->handles();
  std::shared_ptr<const DrafterHandles> handles(drafter, &drafter->device->handles());
  auto exec = std::make_unique<DrafterExecutor>(CreateGpuOps(), std::move(handles),
                                                DrafterTarget{head.table, head.output}, keep, options, faults);
  return std::make_unique<OpsDrafterEngine>(std::move(exec));
}

}  // namespace

std::unique_ptr<DrafterEngine> MakeGpuDrafterEngine(const DrafterWeights& drafter,
                                                    std::shared_ptr<const core::GgufReader> reader,
                                                    const ModelWeights& target, const DrafterOptions& options,
                                                    std::string* error, const DrafterExecFaults& faults) {
  try {
    auto shared = Acquire(drafter, reader, error);
    if (!shared) return nullptr;
    auto target_device = AcquireGpuTargetModel(target, error);
    if (!target_device) return nullptr;
    return Build(shared, std::move(target_device), nullptr, options, faults);
  } catch (const std::exception& e) {
    if (error != nullptr) *error = std::string("GPU drafter: ") + e.what();
  }
  return nullptr;
}

std::unique_ptr<DrafterEngine> MakeGpuDrafterEngineWithHead(const DrafterWeights& drafter,
                                                            std::shared_ptr<const core::GgufReader> reader,
                                                            const ModelWeights& target,
                                                            std::shared_ptr<const core::GgufReader> target_reader,
                                                            const DrafterOptions& options, std::string* error,
                                                            const DrafterExecFaults& faults) {
  try {
    auto shared = Acquire(drafter, reader, error);
    if (!shared) return nullptr;
    auto head = DeviceModel::UploadHead(target, *target_reader, error);
    if (!head) return nullptr;
    return Build(shared, std::shared_ptr<const DeviceModel>(std::move(head)), std::move(target_reader), options, faults);
  } catch (const std::exception& e) {
    if (error != nullptr) *error = std::string("GPU drafter: ") + e.what();
  }
  return nullptr;
}

std::size_t GpuDrafterWeightBytes(const DrafterWeights& drafter) {
  const std::lock_guard lock(g_mutex);
  const auto it = Slots().find(&drafter);
  if (it == Slots().end()) return 0;
  const auto alive = it->second.model.lock();
  return alive ? alive->device->resident_bytes() : 0;
}

}  // namespace gufo::models::ricursive
