// GPU implementation of the executor's operations (hip/ops.hpp): device
// memory (hipMalloc, as LESSONS.md advises for large long-lived data) and the
// real launch functions of hip/kernels.hpp, hip/project.hpp and hip/embed.hpp
// on one stream per executor. Every operator enqueues and returns; CopyOut,
// EmbedRows and Synchronize wait. A launch that reports failure becomes a
// std::runtime_error carrying the kernel's message. Not run on a GPU yet
// (docs/S2-EXECUTOR.md).
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "hip/device_model.hpp"
#include "hip/executor.hpp"
#include "hip/kernels.hpp"
#include "hip/project.hpp"

namespace gufo::models::ricursive {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("ricursive GPU ops: ") + what + ": " +
                             hipGetErrorString(status));
  }
}

void Launched(bool ok, const std::string& error, const char* what) {
  if (!ok) {
    throw std::runtime_error(std::string("ricursive GPU ops: ") + what +
                             " launch refused: " + error);
  }
}

rocm::KvType Kv(KvType type) {
  return type == KvType::kF16 ? rocm::KvType::kF16 : rocm::KvType::kF32;
}

class GpuOps final : public Ops {
 public:
  GpuOps() { Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "create stream"); }
  ~GpuOps() override {
    if (stream_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
      (void)hipStreamDestroy(stream_);
    }
  }

  const char* Name() const noexcept override { return "gpu"; }

  void* Allocate(std::size_t bytes) override {
    void* ptr = nullptr;
    Check(hipMalloc(&ptr, bytes == 0 ? 1 : bytes), "hipMalloc");
    return ptr;
  }
  void Free(void* ptr) noexcept override {
    if (ptr != nullptr) (void)hipFree(ptr);
  }
  void Memset(void* dst, int byte, std::size_t bytes) override {
    Check(hipMemsetAsync(dst, byte, bytes, stream_), "memset");
  }
  void CopyIn(void* dst, const void* host_src, std::size_t bytes) override {
    Check(hipMemcpyAsync(dst, host_src, bytes, hipMemcpyHostToDevice, stream_), "copy in");
    Check(hipStreamSynchronize(stream_), "copy in sync");  // the host source is reusable at once
  }
  void CopyOut(void* host_dst, const void* src, std::size_t bytes) override {
    Check(hipMemcpyAsync(host_dst, src, bytes, hipMemcpyDeviceToHost, stream_), "copy out");
    Check(hipStreamSynchronize(stream_), "copy out sync");
  }
  void CopyInternal(void* dst, const void* src, std::size_t bytes) override {
    Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream_), "device copy");
  }
  void Synchronize() override { Check(hipStreamSynchronize(stream_), "stream sync"); }
  bool Usable() noexcept override {
    return hipStreamSynchronize(stream_) == hipSuccess && hipGetLastError() == hipSuccess;
  }
  void ReserveProjection(std::size_t rows, std::size_t cols) override {
    scratch_.Reserve(rows, cols);
  }

  void EmbedRows(const EmbeddingTable& table, std::span<const std::int32_t> tokens,
                 float scale, float* out) override {
    rocm::EmbedTokens(table, tokens, scale, out, stream_, staging_);
  }
  void ScaleRows(const float* in, float* out, std::uint32_t rows, std::uint32_t dim,
                 float scale) override {
    std::string e;
    Launched(rocm::ScaleRows(in, out, rows, dim, scale, stream_, &e), e, "ScaleRows");
  }
  void RmsNorm(const float* x, const float* weight, float* out, std::uint32_t rows,
               std::uint32_t dim, float eps) override {
    std::string e;
    Launched(rocm::RmsNorm(x, weight, out, rows, dim, eps, stream_, &e), e, "RmsNorm");
  }
  void ResidualPostNorm(float* x, const float* branch, const float* weight,
                        std::uint32_t rows, std::uint32_t dim, float eps) override {
    std::string e;
    Launched(rocm::ResidualPostNorm(x, branch, weight, rows, dim, eps, stream_, &e), e,
             "ResidualPostNorm");
  }
  void ValueMix(const float* x0, const float* a, const float* lambda, float* out,
                std::uint32_t rows, std::uint32_t dim) override {
    std::string e;
    Launched(rocm::ValueMix(x0, a, lambda, out, rows, dim, stream_, &e), e, "ValueMix");
  }
  void Project(const WeightView& w, const float* x, int rows, float* y, bool exact) override {
    const rocm::DeviceTensor t{w.data, w.type, w.rows, w.cols, w.bytes};
    rocm::Project(t, x, rows, y, scratch_, stream_, exact);  // throws on refusal
  }
  void QkNormRopeAppend(float* q, const float* k, const float* v, const float* q_gamma,
                        const float* k_gamma, void* k_cache, void* v_cache, KvType type,
                        std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv_heads,
                        float eps, std::uint32_t start_pos, std::uint32_t capacity,
                        std::uint32_t rotary_dim, float theta) override {
    std::string e;
    Launched(rocm::QkNormRopeAppend(q, k, v, q_gamma, k_gamma, k_cache, v_cache, Kv(type),
                                    tokens, heads, kv_heads, eps, start_pos, capacity,
                                    rotary_dim, theta, stream_, &e),
             e, "QkNormRopeAppend");
  }
  void Attention(const float* q, const void* k_cache, const void* v_cache, KvType type,
                 float* out, std::uint32_t tokens, std::uint32_t heads,
                 std::uint32_t kv_heads, std::uint32_t window, std::uint32_t start_pos,
                 std::uint32_t capacity, float scale) override {
    std::string e;
    Launched(rocm::Attention128(q, k_cache, v_cache, Kv(type), out, tokens, heads, kv_heads,
                                window, start_pos, capacity, scale, stream_, &e),
             e, "Attention128");
  }
  void SigmoidGate(const float* attn, const float* gate, float* out, std::uint32_t rows,
                   std::uint32_t dim) override {
    std::string e;
    Launched(rocm::SigmoidGate(attn, gate, out, rows, dim, stream_, &e), e, "SigmoidGate");
  }
  void ClampedSwiGlu(const float* gate, const float* up, float* out, std::uint32_t rows,
                     std::uint32_t dim, float clamp) override {
    std::string e;
    Launched(rocm::ClampedSwiGlu(gate, up, out, rows, dim, clamp, stream_, &e), e,
             "ClampedSwiGlu");
  }
  void GatherRows(const float* src, const std::int32_t* ids, float* dst, std::uint32_t n_ids,
                  std::uint32_t src_rows, std::uint32_t dim) override {
    std::string e;
    Launched(rocm::GatherRows(src, ids, dst, n_ids, src_rows, dim, stream_, &e), e,
             "GatherRows");
  }

 private:
  hipStream_t stream_{nullptr};
  rocm::Scratch scratch_;
  rocm::EmbedStaging staging_;
};

}  // namespace

GpuExecutor::GpuExecutor(std::shared_ptr<const DeviceModel> model,
                         const ExecutorConfig& config, const ExecFaults& faults)
    : OpsExecutor(std::make_unique<GpuOps>(), model->handles(), model, config,
                  static_cast<std::uint32_t>(rocm::kSplitRows), faults) {}

}  // namespace gufo::models::ricursive
