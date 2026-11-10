#include "cpu_executor.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>

namespace gufo::models::ricursive {
namespace {

// The layer arithmetic is Reference's own (Embed, Layer with its attention
// callback, Head), so this backend cannot drift from the oracle. Those
// members are private and reference.* is not ours to edit; until it grants
// access (`friend class CpuExecutor;`, see docs/EXECUTOR-CONTRACT.md), their
// addresses are taken through explicit instantiation, which the standard
// exempts from access checking ([temp.spec.general]/6).
template<class Tag, typename Tag::type Member>
struct Expose {
  friend typename Tag::type Get(Tag) { return Member; }
};

using Attend =
    std::function<void(const float*, const float*, const float*, float*)>;
using RowFn = std::function<const float*(std::size_t)>;

struct EmbedTag {
  using type = std::vector<float> (Reference::*)(std::span<const std::int32_t>,
                                                 std::vector<float>*) const;
  friend type Get(EmbedTag);
};
struct LayerTag {
  using type = void (Reference::*)(std::size_t, std::size_t, std::size_t,
                                   std::vector<float>&,
                                   const std::vector<float>&,
                                   std::vector<float>&, const Attend&,
                                   const TapFn&) const;
  friend type Get(LayerTag);
};
struct HeadTag {
  using type = std::vector<float> (Reference::*)(const std::vector<float>&,
                                                 std::size_t, bool,
                                                 const TapFn&) const;
  friend type Get(HeadTag);
};
struct AttendRowTag {
  using type = void (Reference::*)(const float*, std::size_t, std::size_t,
                                   const RowFn&, const RowFn&, float*) const;
  friend type Get(AttendRowTag);
};
struct WindowTag {
  using type = std::size_t (Reference::*)() const noexcept;
  friend type Get(WindowTag);
};

template struct Expose<EmbedTag, &Reference::Embed>;
template struct Expose<LayerTag, &Reference::Layer>;
template struct Expose<HeadTag, &Reference::Head>;
template struct Expose<AttendRowTag, &Reference::AttendRow>;
template struct Expose<WindowTag, &Reference::EffectiveWindow>;

ExecutorInfo CpuInfo(const ModelWeights& weights, const ExecutorConfig& config) {
  if (config.kv_type != KvType::kF32) {
    throw std::invalid_argument("ricursive CPU executor: KV type must be F32");
  }
  // Every row is computed alone (fixed dot order), at any width.
  return ExecutorInfo::Make(weights.config, config, config.max_rows);
}

}  // namespace

CpuExecutor::CpuExecutor(const ModelWeights& weights,
                         const ExecutorConfig& config, Faults faults)
    : Executor(CpuInfo(weights, config), faults.ledger),
      ref_(weights, faults.variant),
      chunk_local_rope_(faults.chunk_local_rope),
      slots_(Info().ring_rows - std::min(faults.ring_shortfall,
                                         Info().ring_rows - 1)) {
  if (faults.variant == Variant::kNonCausal) {
    throw std::invalid_argument(
        "ricursive CPU executor: non_causal needs ForwardFull");
  }
  const ExecutorInfo& info = Info();
  k_.resize(info.layers);
  v_.resize(info.layers);
  for (std::uint32_t l = 0; l < info.layers; ++l) {
    const std::size_t rows = info.sliding[l] ? slots_ : info.max_context;
    k_[l].assign(rows * info.kv_width, 0.0F);
    v_[l].assign(rows * info.kv_width, 0.0F);
  }
}

const float* CpuExecutor::Row(std::uint32_t layer, KvPart part,
                              std::uint32_t position) const {
  const auto& store = part == KvPart::kKey ? k_[layer] : v_[layer];
  const std::size_t slot =
      Info().sliding[layer] ? position % slots_ : position;
  return store.data() + slot * Info().kv_width;
}

float* CpuExecutor::Row(std::uint32_t layer, KvPart part,
                        std::uint32_t position) {
  return const_cast<float*>(std::as_const(*this).Row(layer, part, position));
}

void CpuExecutor::DoForward(std::span<const std::int32_t> tokens,
                            std::uint32_t first_position, LogitRows which,
                            std::span<float> logits, const Tap& tap) {
  const ExecutorInfo& info = Info();
  const Config& c = ref_.weights().config;
  const std::size_t n = tokens.size(), Q = c.QueryWidth(), KV = info.kv_width;
  const std::uint32_t p0 = first_position;
  std::vector<float> x0;
  std::vector<float> h = (ref_.*Get(EmbedTag{}))(tokens, &x0);
  if (tap) tap("embd_scaled", -1, h.data(), n, c.embedding_length);
  std::vector<float> v_layer0;
  const std::size_t window = (ref_.*Get(WindowTag{}))();
  for (std::uint32_t il = 0; il < info.layers; ++il) {
    const bool sliding = info.sliding[il];
    const Attend attend = [&](const float* q, const float* k, const float* v,
                              float* o) {
      // Every row of the chunk is stored before any query reads (as on the
      // GPU), so an undersized ring shows here, not only on the GPU.
      for (std::size_t t = 0; t < n; ++t) {
        const auto p = static_cast<std::uint32_t>(p0 + t);
        std::copy_n(k + t * KV, KV, Row(il, KvPart::kKey, p));
        std::copy_n(v + t * KV, KV, Row(il, KvPart::kValue, p));
      }
      const RowFn key = [&](std::size_t j) {
        return Row(il, KvPart::kKey, static_cast<std::uint32_t>(j));
      };
      const RowFn val = [&](std::size_t j) {
        return Row(il, KvPart::kValue, static_cast<std::uint32_t>(j));
      };
      for (std::size_t t = 0; t < n; ++t) {
        const std::size_t pos = p0 + t;
        const std::size_t lo = sliding && pos + 1 > window ? pos + 1 - window : 0;
        (ref_.*Get(AttendRowTag{}))(q + t * Q, lo, pos, key, val, o + t * Q);
      }
    };
    (ref_.*Get(LayerTag{}))(il, n, chunk_local_rope_ ? 0 : p0, h, x0, v_layer0,
                            attend, tap);
  }
  if (which == LogitRows::kNone) return;
  const auto out = (ref_.*Get(HeadTag{}))(h, n, which == LogitRows::kLast, tap);
  if (out.size() != logits.size()) {
    throw std::logic_error("ricursive CPU executor: logit size mismatch");
  }
  std::copy(out.begin(), out.end(), logits.begin());
}

void CpuExecutor::DoExport(std::uint32_t layer, KvPart part, RowRange rows,
                           std::span<std::byte> out) const {
  const std::size_t bytes = Info().RowBytes();
  for (std::uint32_t p = rows.first; p < rows.end; ++p) {
    std::memcpy(out.data() + std::size_t{p - rows.first} * bytes,
                Row(layer, part, p), bytes);
  }
}

void CpuExecutor::DoImport(std::uint32_t layer, KvPart part, RowRange rows,
                           std::span<const std::byte> bytes) {
  const std::size_t row = Info().RowBytes();
  for (std::uint32_t p = rows.first; p < rows.end; ++p) {
    std::memcpy(Row(layer, part, p),
                bytes.data() + std::size_t{p - rows.first} * row, row);
  }
}

}  // namespace gufo::models::ricursive
