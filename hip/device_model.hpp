#ifndef RICURSIVE_HIP_DEVICE_MODEL_HPP_
#define RICURSIVE_HIP_DEVICE_MODEL_HPP_

// The model's weights resident in device memory, still in their GGUF encoding
// (the Gemma 4 extension's Uploader pattern: public WeightUpload and
// WeightAllocator, every allocation followed by a zeroed kWeightTailBytes
// margin because the quantized kernels over-read the last row). The token
// table is NOT uploaded: IQ4_XS has no GPU path, so embedding rows are looked
// up on the host from the mapped GGUF. Needs no HIP header.

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <string>
#include <vector>

#include "hip/drafter_handles.hpp"
#include "hip/ops.hpp"
#include "src/drafter_weights.hpp"
#include "src/weights.hpp"

namespace gufo::models::ricursive {

class DeviceModel {
 public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Copies every projection matrix and F32 vector to the device. `reader`
  /// must be the reader `weights` was resolved from and outlive the model.
  /// Returns nullptr and sets `*error` on failure (unsupported type for a
  /// projection, allocation or copy failure); nothing stays allocated then.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error);

  /// Only what the drafter shares: the output head and its norm (device), the token table (host
  /// mapping). handles().layers is empty: no executor of the target may use this model. For tools that
  /// run the drafter alone (tools/drafter_bench.cpp) without the 40 GB of the target's layers.
  [[nodiscard]] static std::unique_ptr<DeviceModel> UploadHead(
      const ModelWeights& weights, const core::GgufReader& reader,
      std::string* error);

  [[nodiscard]] const ModelHandles& handles() const noexcept { return handles_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

 private:
  DeviceModel() = default;
  ModelHandles handles_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

/// The drafter's weights resident in device memory (same encoding and tail margin as DeviceModel).
class DeviceDrafter {
 public:
  ~DeviceDrafter();
  DeviceDrafter(const DeviceDrafter&) = delete;
  DeviceDrafter& operator=(const DeviceDrafter&) = delete;

  /// `reader` must be the reader `drafter` was resolved from and outlive the object. Returns
  /// nullptr and sets `*error` on failure; nothing stays allocated then.
  [[nodiscard]] static std::unique_ptr<DeviceDrafter> Upload(
      const DrafterWeights& drafter, const core::GgufReader& reader,
      std::string* error);

  [[nodiscard]] const DrafterHandles& handles() const noexcept { return handles_; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

 private:
  DeviceDrafter() = default;
  DrafterHandles handles_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_DEVICE_MODEL_HPP_
