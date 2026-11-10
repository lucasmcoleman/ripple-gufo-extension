// Order-preserving JSON value with Python-compatible parsing and serialisation.
//
// Dependency-free (standard library only). Written for the Ripple chat
// template, whose `tojson` filter is Python's json.dumps(ensure_ascii=False)
// with default separators and preserved key order.
#ifndef RICURSIVE_TEXT_JSON_VALUE_HPP_
#define RICURSIVE_TEXT_JSON_VALUE_HPP_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ricursive::text {

class JsonValue {
public:
  enum class Type : std::uint8_t {
    kNull, kBool, kInt, kDouble, kString, kArray, kObject
  };
  using Array = std::vector<JsonValue>;
  using Member = std::pair<std::string, JsonValue>;
  using Object = std::vector<Member>;

  JsonValue() = default;  // null
  static JsonValue Bool(bool v);
  static JsonValue Int(std::int64_t v);
  /// Arbitrary-size integer given as a canonical decimal string ("-12", "0").
  static JsonValue BigInt(std::string decimal);
  static JsonValue Double(double v);
  static JsonValue String(std::string v);
  static JsonValue MakeArray(Array v = {});
  static JsonValue MakeObject(Object v = {});

  [[nodiscard]] Type type() const noexcept { return type_; }
  [[nodiscard]] bool is_null() const noexcept { return type_ == Type::kNull; }
  [[nodiscard]] bool is_string() const noexcept { return type_ == Type::kString; }
  [[nodiscard]] bool is_array() const noexcept { return type_ == Type::kArray; }
  [[nodiscard]] bool is_object() const noexcept { return type_ == Type::kObject; }

  [[nodiscard]] bool bool_value() const noexcept { return bool_; }
  [[nodiscard]] double double_value() const noexcept { return double_; }
  /// String payload for kString, canonical decimal text for kInt.
  [[nodiscard]] const std::string& text() const noexcept { return text_; }
  [[nodiscard]] const Array& array() const noexcept { return array_; }
  [[nodiscard]] Array& array() noexcept { return array_; }
  [[nodiscard]] const Object& object() const noexcept { return object_; }
  [[nodiscard]] Object& object() noexcept { return object_; }

  /// Object member lookup; nullptr when absent or not an object.
  [[nodiscard]] const JsonValue* Find(std::string_view key) const noexcept;
  /// Replace in place if the key exists (position kept), else append.
  void Set(std::string key, JsonValue value);

  /// Python truthiness: null/false/0/0.0/""/[]/{} are falsy.
  [[nodiscard]] bool Truthy() const noexcept;

private:
  Type type_{Type::kNull};
  bool bool_{false};
  double double_{0};
  std::string text_;
  Array array_;
  Object object_;
};

/// Parses like Python's json.loads (strict): duplicate keys keep the last value
/// at the first position, NaN/Infinity/-Infinity accepted, integers keep full
/// precision. Lone \u surrogate escapes become U+FFFD (Python would keep an
/// unencodable lone surrogate). Returns false and sets *error on failure.
[[nodiscard]] bool ParseJson(std::string_view text, JsonValue* out,
                             std::string* error = nullptr);

/// Python json.dumps(value, ensure_ascii=False) with default separators.
[[nodiscard]] std::string DumpPythonJson(const JsonValue& value);
void AppendPythonJson(std::string& out, const JsonValue& value);

}  // namespace ricursive::text

#endif  // RICURSIVE_TEXT_JSON_VALUE_HPP_
