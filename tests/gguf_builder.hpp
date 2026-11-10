#ifndef RICURSIVE_TESTS_GGUF_BUILDER_HPP_
#define RICURSIVE_TESTS_GGUF_BUILDER_HPP_

// Serializes a GGUF v3 image in memory. Structure derived from the MIT-licensed
// Gemma 4 extension's tests/gguf_builder.hpp (see THIRD_PARTY.md).

#include <cstdint>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ricursive_test {

struct BoolArray {
  std::vector<bool> values;
};
struct U32Array {
  std::vector<std::uint32_t> values;
};
struct F32Array {
  std::vector<float> values;
};

using MetaValue = std::variant<std::uint32_t, float, bool, std::string,
                               BoolArray, U32Array, F32Array>;
using Metadata = std::map<std::string, MetaValue>;

struct TensorSpec {
  std::vector<std::uint64_t> dims;
  gufo::core::GgmlType type;
  std::vector<std::uint8_t> data;
};
using Tensors = std::map<std::string, TensorSpec>;

class GgufImage {
 public:
  GgufImage(const Metadata& metadata, const Tensors& tensors = {}) {
    Build(metadata, tensors);
    reader_ = gufo::core::GgufReader::OpenMemory(bytes_.data(), bytes_.size(),
                                                 &error_);
  }
  [[nodiscard]] const gufo::core::GgufReader* reader() const {
    return reader_.get();
  }
  [[nodiscard]] const std::string& error() const { return error_; }

 private:
  template<class T>
  void Pod(T value) {
    const auto begin = bytes_.size();
    bytes_.resize(begin + sizeof(value));
    std::memcpy(bytes_.data() + begin, &value, sizeof(value));
  }
  void Text(const std::string& value) {
    Pod(std::uint64_t{value.size()});
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void Align() { bytes_.resize((bytes_.size() + 31) / 32 * 32); }

  void Build(const Metadata& metadata, const Tensors& tensors) {
    Pod(std::uint32_t{0x46554747});
    Pod(std::uint32_t{3});
    Pod(std::uint64_t{tensors.size()});
    Pod(std::uint64_t{metadata.size()});
    for (const auto& [key, value] : metadata) {
      Text(key);
      std::visit(
          [&](const auto& item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::uint32_t>) {
              Pod(std::uint32_t{4});
              Pod(item);
            } else if constexpr (std::is_same_v<T, float>) {
              Pod(std::uint32_t{6});
              Pod(item);
            } else if constexpr (std::is_same_v<T, bool>) {
              Pod(std::uint32_t{7});
              Pod(static_cast<std::uint8_t>(item ? 1 : 0));
            } else if constexpr (std::is_same_v<T, std::string>) {
              Pod(std::uint32_t{8});
              Text(item);
            } else if constexpr (std::is_same_v<T, BoolArray>) {
              Pod(std::uint32_t{9});
              Pod(std::uint32_t{7});
              Pod(std::uint64_t{item.values.size()});
              for (bool v : item.values) Pod(static_cast<std::uint8_t>(v ? 1 : 0));
            } else if constexpr (std::is_same_v<T, U32Array>) {
              Pod(std::uint32_t{9});
              Pod(std::uint32_t{4});
              Pod(std::uint64_t{item.values.size()});
              for (auto v : item.values) Pod(v);
            } else {
              static_assert(std::is_same_v<T, F32Array>);
              Pod(std::uint32_t{9});
              Pod(std::uint32_t{6});
              Pod(std::uint64_t{item.values.size()});
              for (auto v : item.values) Pod(v);
            }
          },
          value);
    }
    std::uint64_t offset = 0;
    for (const auto& [name, spec] : tensors) {
      Text(name);
      Pod(static_cast<std::uint32_t>(spec.dims.size()));
      for (auto d : spec.dims) Pod(d);
      Pod(static_cast<std::uint32_t>(spec.type));
      Pod(offset);
      offset += (spec.data.size() + 31) / 32 * 32;
    }
    Align();
    for (const auto& [name, spec] : tensors) {
      (void)name;
      bytes_.insert(bytes_.end(), spec.data.begin(), spec.data.end());
      Align();
    }
    if (tensors.empty()) Align();
  }

  std::vector<std::uint8_t> bytes_;
  std::unique_ptr<gufo::core::GgufReader> reader_;
  std::string error_;
};

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_GGUF_BUILDER_HPP_
