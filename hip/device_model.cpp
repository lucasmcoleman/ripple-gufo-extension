#include "hip/device_model.hpp"

#include <hip/hip_runtime.h>

#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>

#include "hip/project.hpp"

namespace gufo::models::ricursive {
namespace {

struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator& placement;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) *error = message;
    ok = false;
  }

  void* Allocate(std::size_t size, const std::string& what) {
    void* ptr = nullptr;
    if (placement.Allocate(&ptr, size + rocm::kWeightTailBytes) != hipSuccess ||
        ptr == nullptr) {
      Fail("device allocation failed for " + what + " (" + std::to_string(size) +
           " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size + rocm::kWeightTailBytes;
    // Zeroed tail: the K-quant kernels read whole 256-element iterations.
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                         rocm::kWeightTailBytes, nullptr);
    return ptr;
  }

  /// Queues the tensor's bytes; the device pointer, or nullptr after Fail.
  void* Copy(const TensorRef& t) {
    if (!ok) return nullptr;
    if (t.empty()) {
      Fail("tensor " + std::string(t.name) + " has no data");
      return nullptr;
    }
    const std::size_t size = t.RowBytes() * t.rows;
    void* ptr = Allocate(size, std::string(t.name));
    if (ptr == nullptr) return nullptr;
    if (!stager.CopyMapped(t.data, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name));
      return nullptr;
    }
    return ptr;
  }

  WeightView Matrix(const TensorRef& t) {
    WeightView v;
    if (!rocm::IsProjectable(t.type)) {
      Fail("tensor " + std::string(t.name) + " has type " +
           std::string(core::ToString(t.type)) + ", which has no GPU projection path");
      return v;
    }
    v.type = t.type;
    v.rows = static_cast<std::uint32_t>(t.rows);
    v.cols = static_cast<std::uint32_t>(t.cols);
    v.bytes = t.RowBytes() * t.rows;
    if (rocm::RowBytes(t.type, t.cols) * t.rows != v.bytes) {
      Fail("tensor " + std::string(t.name) + ": size does not match its shape");
      return v;
    }
    v.data = Copy(t);
    return v;
  }

  const float* Vector(const TensorRef& t) {
    if (t.type != core::GgmlType::kF32) {
      Fail("tensor " + std::string(t.name) + " must be F32");
      return nullptr;
    }
    return static_cast<const float*>(Copy(t));
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* ptr : allocations_) (void)hipFree(ptr);
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& weights,
                                                 const core::GgufReader& reader,
                                                 std::string* error) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  const auto regions = reader.GetMappedRegions();
  const std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error);
  if (!stager) return nullptr;
  std::uint64_t file_bytes = 0;
  for (const auto& shard : shards) file_bytes += shard.size;
  const hip::WeightAllocator placement(file_bytes);
  Uploader up{*stager, placement, m->allocations_, m->bytes_, error};

  ModelHandles& h = m->handles_;
  h.config = weights.config;
  h.table = EmbeddingTable{weights.token_embd.data, weights.token_embd.type,
                           static_cast<std::size_t>(weights.token_embd.rows),
                           static_cast<std::size_t>(weights.token_embd.cols)};
  h.output = up.Matrix(weights.output);
  h.output_norm = up.Vector(weights.output_norm);
  for (const LayerWeights& l : weights.layers) {
    LayerHandles d;
    d.attn_q = up.Matrix(l.attn_q);
    d.attn_k = up.Matrix(l.attn_k);
    d.attn_v = up.Matrix(l.attn_v);
    d.attn_output = up.Matrix(l.attn_output);
    d.attn_gate = up.Matrix(l.attn_gate);
    d.ffn_gate = up.Matrix(l.ffn_gate);
    d.ffn_up = up.Matrix(l.ffn_up);
    d.ffn_down = up.Matrix(l.ffn_down);
    d.attn_norm = up.Vector(l.attn_norm);
    d.attn_q_norm = up.Vector(l.attn_q_norm);
    d.attn_k_norm = up.Vector(l.attn_k_norm);
    d.post_attention_norm = up.Vector(l.post_attention_norm);
    d.ffn_norm = up.Vector(l.ffn_norm);
    d.post_ffw_norm = up.Vector(l.post_ffw_norm);
    d.attn_vres_lambda = up.Vector(l.attn_vres_lambda);
    h.layers.push_back(d);
    if (!up.ok) break;
  }
  // Finish waits for every queued copy, on failure too.
  std::string finish_error;
  const bool finished = stager->Finish(&finish_error);
  if (!up.ok) return nullptr;
  if (!finished) {
    if (error != nullptr) *error = finish_error;
    return nullptr;
  }
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error != nullptr)
      *error = std::string("weight upload failed: ") + hipGetErrorString(status);
    return nullptr;
  }
  return m;
}


std::unique_ptr<DeviceModel> DeviceModel::UploadHead(const ModelWeights& weights, const core::GgufReader& reader,
                                                     std::string* error) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  const auto regions = reader.GetMappedRegions();
  const std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error);
  if (!stager) return nullptr;
  std::uint64_t file_bytes = 0;
  for (const auto& shard : shards) file_bytes += shard.size;
  const hip::WeightAllocator placement(file_bytes);
  Uploader up{*stager, placement, m->allocations_, m->bytes_, error};
  ModelHandles& h = m->handles_;
  h.config = weights.config;
  h.table = EmbeddingTable{weights.token_embd.data, weights.token_embd.type,
                           static_cast<std::size_t>(weights.token_embd.rows),
                           static_cast<std::size_t>(weights.token_embd.cols)};
  h.output = up.Matrix(weights.output);
  h.output_norm = up.Vector(weights.output_norm);
  std::string finish_error;
  const bool finished = stager->Finish(&finish_error);
  if (!up.ok) return nullptr;
  if (!finished) {
    if (error != nullptr) *error = finish_error;
    return nullptr;
  }
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error != nullptr) *error = std::string("weight upload failed: ") + hipGetErrorString(status);
    return nullptr;
  }
  return m;
}

DeviceDrafter::~DeviceDrafter() {
  for (void* ptr : allocations_) (void)hipFree(ptr);
}

std::unique_ptr<DeviceDrafter> DeviceDrafter::Upload(const DrafterWeights& w, const core::GgufReader& reader,
                                                     std::string* error) {
  std::unique_ptr<DeviceDrafter> m(new DeviceDrafter());
  const auto regions = reader.GetMappedRegions();
  const std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error);
  if (!stager) return nullptr;
  std::uint64_t file_bytes = 0;
  for (const auto& shard : shards) file_bytes += shard.size;
  const hip::WeightAllocator placement(file_bytes);
  Uploader up{*stager, placement, m->allocations_, m->bytes_, error};
  DrafterHandles& h = m->handles_;
  h.config = w.config;
  h.fc = up.Matrix(w.fc);
  h.selector_hidden = up.Matrix(w.selector_hidden);
  h.selector_predecessor = up.Matrix(w.selector_predecessor);
  h.selector_successor = up.Matrix(w.selector_successor);
  h.enc_output_norm = up.Vector(w.enc_output_norm);
  h.output_norm = up.Vector(w.output_norm);
  for (const DrafterLayerWeights& l : w.layers) {
    DrafterLayerHandles d;
    d.attn_q = up.Matrix(l.attn_q);
    d.attn_k = up.Matrix(l.attn_k);
    d.attn_v = up.Matrix(l.attn_v);
    d.attn_output = up.Matrix(l.attn_output);
    d.attn_conv_proj = up.Matrix(l.attn_conv_proj);
    d.ffn_gate = up.Matrix(l.ffn_gate);
    d.ffn_up = up.Matrix(l.ffn_up);
    d.ffn_down = up.Matrix(l.ffn_down);
    d.ffn_conv_proj = up.Matrix(l.ffn_conv_proj);
    d.attn_norm = up.Vector(l.attn_norm);
    d.attn_q_norm = up.Vector(l.attn_q_norm);
    d.attn_k_norm = up.Vector(l.attn_k_norm);
    d.attn_conv_base = up.Vector(l.attn_conv_base);
    d.ffn_norm = up.Vector(l.ffn_norm);
    d.ffn_conv_base = up.Vector(l.ffn_conv_base);
    h.layers.push_back(d);
    if (!up.ok) break;
  }
  std::string finish_error;
  const bool finished = stager->Finish(&finish_error);
  if (!up.ok) return nullptr;
  if (!finished) {
    if (error != nullptr) *error = finish_error;
    return nullptr;
  }
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error != nullptr) *error = std::string("weight upload failed: ") + hipGetErrorString(status);
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::ricursive
