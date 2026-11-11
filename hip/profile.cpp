#include "hip/profile.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <utility>

namespace gufo::models::ricursive::profile {
namespace {

constexpr const char* kLayerNames[kLayerStageCount] = {
    "attn_norm", "value_mix", "proj_q",       "proj_k",     "proj_gate",
    "proj_v",    "qknorm_rope_kv", "attention", "sigmoid_gate", "proj_attn_out",
    "post_attn_residual", "ffn_norm", "proj_ffn_gate", "proj_ffn_up", "swiglu",
    "proj_ffn_down", "post_ffn_residual"};
constexpr const char* kPassNames[kPassStageCount] = {
    "embed", "embed_scale", "row_gather", "final_norm", "output_proj", "logits_readback"};

std::string Fmt(const char* format, ...) __attribute__((format(printf, 1, 2)));
std::string Fmt(const char* format, ...) {
  char buf[256];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);
  return buf;
}

}  // namespace

std::string StageName(StageId id) {
  if (id < kLayerStageCount) return std::string("sw.") + kLayerNames[id];
  if (id < 2 * kLayerStageCount) return std::string("gl.") + kLayerNames[id - kLayerStageCount];
  if (id < kStageCount) return kPassNames[id - 2 * kLayerStageCount];
  return "?";
}

bool IsMatmul(StageId id) noexcept {
  if (id >= 2 * kLayerStageCount) return id == Id(PassStage::kOutputProj);
  switch (static_cast<LayerStage>(id % kLayerStageCount)) {
    case LayerStage::kProjQ:
    case LayerStage::kProjK:
    case LayerStage::kProjGate:
    case LayerStage::kProjV:
    case LayerStage::kProjAttnOut:
    case LayerStage::kProjFfnGate:
    case LayerStage::kProjFfnUp:
    case LayerStage::kProjFfnDown:
      return true;
    default:
      return false;
  }
}

Bucket BucketFor(std::uint32_t rows, std::uint32_t context_len) noexcept {
  if (rows != 1) return Bucket::kPrompt;
  if (context_len < 1024) return Bucket::kGenLt1024;
  if (context_len < 2048) return Bucket::kGen1024;
  if (context_len < 4096) return Bucket::kGen2048;
  return Bucket::kGen4096;
}

const char* BucketTag(Bucket b) noexcept {
  switch (b) {
    case Bucket::kPrompt: return "prompt";
    case Bucket::kGenLt1024: return "gen_lt1024";
    case Bucket::kGen1024: return "gen_1024_2047";
    case Bucket::kGen2048: return "gen_2048_4095";
    case Bucket::kGen4096: return "gen_ge4096";
  }
  return "?";
}

std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void StageProfile::BeginPass(std::uint32_t rows, std::uint32_t context_len) {
  cur_ = BucketStats{};
  cur_bucket_ = BucketFor(rows, context_len);
  cur_rows_ = rows;
  open_ = true;
  pass_start_ = last_ = clock_();
}

void StageProfile::Mark(StageId id, std::uint64_t weight_bytes) {
  if (!open_ || id >= kStageCount) return;
  const std::int64_t now = clock_();
  cur_.ns[id] += now - last_;
  cur_.bytes[id] += weight_bytes;
  ++cur_.syncs;
  last_ = now;
}

void StageProfile::EndPass() {
  if (!open_) return;
  open_ = false;
  BucketStats& b = buckets_[static_cast<std::size_t>(cur_bucket_)];
  const std::int64_t end = clock_();
  ++b.passes;
  b.rows += cur_rows_;
  b.syncs += cur_.syncs;
  b.total_ns += end - pass_start_;
  for (std::size_t i = 0; i < kStageCount; ++i) {
    b.ns[i] += cur_.ns[i];
    b.bytes[i] += cur_.bytes[i];
  }
}

std::uint64_t StageProfile::TotalPasses() const noexcept {
  std::uint64_t n = 0;
  for (const BucketStats& b : buckets_) n += b.passes;
  return n;
}

std::string StageProfile::Report() const {
  std::string out;
  for (std::size_t bi = 0; bi < kBucketCount; ++bi) {
    const BucketStats& b = buckets_[bi];
    if (b.passes == 0) continue;
    const char* tag = BucketTag(static_cast<Bucket>(bi));
    const double passes = static_cast<double>(b.passes);
    const double total_ms = static_cast<double>(b.total_ns) / 1e6;
    // One line per row of the table; the first column is the bucket so the
    // whole report greps per bucket.
    const auto line = [&](const std::string& name, std::int64_t ns, std::uint64_t bytes,
                          bool weights) {
      const double ms = static_cast<double>(ns) / 1e6;
      const double share = b.total_ns > 0 ? 100.0 * static_cast<double>(ns) / static_cast<double>(b.total_ns) : 0.0;
      out += Fmt("%-14s %-20s %11.3f %10.4f %7.2f", tag, name.c_str(), ms, ms / passes, share);
      if (weights && bytes > 0) {
        const double gbps = ns > 0 ? static_cast<double>(bytes) / static_cast<double>(ns) : 0.0;
        out += Fmt(" %11.2f %8.1f\n", static_cast<double>(bytes) / passes / 1e6, gbps);
      } else {
        out += Fmt(" %11s %8s\n", "-", "-");
      }
    };
    out += Fmt("# bucket %s: %llu passes, %llu rows (%s)\n", tag,
               static_cast<unsigned long long>(b.passes), static_cast<unsigned long long>(b.rows),
               bi == 0 ? "rows > 1" : "1 row, context = tokens including this one");
    out += Fmt("%-14s %-20s %11s %10s %7s %11s %8s\n", "bucket", "stage", "total_ms", "mean_ms",
               "share%", "weight_MB", "GB/s");
    std::int64_t sum_sw = 0, sum_gl = 0, sum_pass = 0, mm_ns = 0;
    std::uint64_t mm_bytes = 0;
    for (StageId id = 0; id < kStageCount; ++id) {
      if (b.ns[id] == 0 && b.bytes[id] == 0) continue;
      line(StageName(id), b.ns[id], b.bytes[id], IsMatmul(id));
      (id < kLayerStageCount ? sum_sw : id < 2 * kLayerStageCount ? sum_gl : sum_pass) += b.ns[id];
      if (IsMatmul(id)) {
        mm_ns += b.ns[id];
        mm_bytes += b.bytes[id];
      }
    }
    line("sw.sum", sum_sw, 0, false);
    line("gl.sum", sum_gl, 0, false);
    line("pass.sum", sum_pass, 0, false);
    line("matmul.all", mm_ns, mm_bytes, true);
    line("unattributed", b.total_ns - sum_sw - sum_gl - sum_pass, 0, false);
    line("pass.total", b.total_ns, 0, false);
    const double seconds = static_cast<double>(b.total_ns) / 1e9;
    out += Fmt("%-14s mean_ms_per_pass %.3f\n", tag, total_ms / passes);
    out += Fmt("%-14s tokens_per_s %.3f\n", tag,
               seconds > 0 ? static_cast<double>(b.rows) / seconds : 0.0);
    out += Fmt("%-14s sync_points_per_pass %.1f\n", tag, static_cast<double>(b.syncs) / passes);
  }
  return out;
}

}  // namespace gufo::models::ricursive::profile
