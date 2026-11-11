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
#include "hip/drafter_kernels.hpp"
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

  // ---- the drafter's operations (hip/drafter_kernels.hip.cpp) ------------
  void DraftQkNormRope(float* q, const float* k, const float* v, const float* q_gamma,
                       const float* k_gamma, void* k_store, void* v_store, KvType type,
                       std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv_heads,
                       std::uint32_t head_dim, float eps, std::uint32_t start_pos,
                       std::uint32_t slot_start, std::uint32_t capacity, bool rope,
                       float theta) override {
    std::string e;
    Launched(rocm::DraftQkNormRope(q, k, v, q_gamma, k_gamma, k_store, v_store, Kv(type), tokens,
                                   heads, kv_heads, head_dim, eps, start_pos, slot_start, capacity,
                                   rope, theta, stream_, &e),
             e, "DraftQkNormRope");
  }
  void DraftAttention(const float* q, const void* k_ctx, const void* v_ctx, KvType type,
                      std::uint32_t ctx_capacity, std::uint32_t ctx_first, std::uint32_t anchor,
                      const float* k_blk, const float* v_blk, float* out, std::uint32_t tokens,
                      std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                      std::uint32_t window, float scale) override {
    std::string e;
    Launched(rocm::DraftAttention(q, k_ctx, v_ctx, Kv(type), ctx_capacity, ctx_first, anchor, k_blk,
                                  v_blk, out, tokens, heads, kv_heads, head_dim, window, scale,
                                  stream_, &e),
             e, "DraftAttention");
  }
  void DynamicConv(const float* input, const float* dynamic_coefficients, const float* base_kernel,
                   float* output, std::uint32_t num_tokens, std::uint32_t hidden_size,
                   std::uint32_t kernel_size, std::uint32_t group_size, std::uint32_t side,
                   bool accumulate) override {
    std::string e;
    Launched(rocm::DraftDynamicConv(input, dynamic_coefficients, base_kernel, output, num_tokens,
                                    hidden_size, kernel_size, group_size, side, accumulate, stream_, &e),
             e, "DraftDynamicConv");
  }
  void SiluMul(float* gate, const float* up, std::uint32_t total_elements) override {
    std::string e;
    Launched(rocm::DraftSiluMul(gate, up, total_elements, stream_, &e), e, "DraftSiluMul");
  }
  void DraftSelect(const float* logits, const float* projected_hidden,
                   const WeightView& predecessor_codebook, const WeightView& successor_codebook,
                   std::int32_t* tokens, float* candidate_scores, std::int32_t* candidate_ids,
                   float* partial_scores, std::int32_t* partial_ids, std::uint32_t rows,
                   std::uint32_t vocab_size, std::uint32_t selector_rank,
                   std::uint32_t selector_top_k) override {
    for (const WeightView* w : {&predecessor_codebook, &successor_codebook}) {
      if (w->data == nullptr || w->cols != selector_rank || w->rows < vocab_size)
        throw std::runtime_error("ricursive GPU ops: DraftSelect: codebook shape");
    }
    std::string e;
    Launched(rocm::DraftSelect(logits, projected_hidden, predecessor_codebook.data,
                               EncodingOf(predecessor_codebook.type), successor_codebook.data,
                               EncodingOf(successor_codebook.type), tokens, candidate_scores,
                               candidate_ids, partial_scores, partial_ids, rows, vocab_size,
                               selector_rank, selector_top_k, stream_, &e),
             e, "DraftSelect");
  }

 private:
  hipStream_t stream_{nullptr};
  rocm::Scratch scratch_;
  rocm::EmbedStaging staging_;
};

}  // namespace

std::unique_ptr<Ops> CreateGpuOps() { return std::make_unique<GpuOps>(); }

GpuExecutor::GpuExecutor(std::shared_ptr<const DeviceModel> model,
                         const ExecutorConfig& config, const ExecFaults& faults)
    : OpsExecutor(std::make_unique<GpuOps>(), model->handles(), model, config,
                  static_cast<std::uint32_t>(rocm::kSplitRows), faults) {}

}  // namespace gufo::models::ricursive
