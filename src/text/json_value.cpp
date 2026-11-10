#include "json_value.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ricursive::text {

JsonValue JsonValue::Bool(bool v) {
  JsonValue j; j.type_ = Type::kBool; j.bool_ = v; return j;
}
JsonValue JsonValue::Int(std::int64_t v) { return BigInt(std::to_string(v)); }
JsonValue JsonValue::BigInt(std::string decimal) {
  JsonValue j; j.type_ = Type::kInt; j.text_ = std::move(decimal); return j;
}
JsonValue JsonValue::Double(double v) {
  JsonValue j; j.type_ = Type::kDouble; j.double_ = v; return j;
}
JsonValue JsonValue::String(std::string v) {
  JsonValue j; j.type_ = Type::kString; j.text_ = std::move(v); return j;
}
JsonValue JsonValue::MakeArray(Array v) {
  JsonValue j; j.type_ = Type::kArray; j.array_ = std::move(v); return j;
}
JsonValue JsonValue::MakeObject(Object v) {
  JsonValue j; j.type_ = Type::kObject; j.object_ = std::move(v); return j;
}

const JsonValue* JsonValue::Find(std::string_view key) const noexcept {
  if (type_ != Type::kObject) return nullptr;
  for (const auto& m : object_)
    if (m.first == key) return &m.second;
  return nullptr;
}

void JsonValue::Set(std::string key, JsonValue value) {
  for (auto& m : object_)
    if (m.first == key) { m.second = std::move(value); return; }
  object_.emplace_back(std::move(key), std::move(value));
}

bool JsonValue::Truthy() const noexcept {
  switch (type_) {
    case Type::kNull: return false;
    case Type::kBool: return bool_;
    case Type::kInt: return text_ != "0";
    case Type::kDouble: return double_ != 0.0;  // NaN is truthy, as in Python
    case Type::kString: return !text_.empty();
    case Type::kArray: return !array_.empty();
    case Type::kObject: return !object_.empty();
  }
  return false;
}

// ------------------------------------------------------------------ parse
namespace {

struct Parser {
  std::string_view s;
  std::size_t i{0};
  std::string err;
  int depth{0};

  bool Fail(const char* m) {
    if (err.empty()) err = std::string(m) + " at byte " + std::to_string(i);
    return false;
  }
  void Ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  static void PutUtf8(std::string& o, std::uint32_t c) {
    if (c < 0x80) o.push_back(char(c));
    else if (c < 0x800) { o.push_back(char(0xC0 | (c >> 6))); o.push_back(char(0x80 | (c & 63))); }
    else if (c < 0x10000) { o.push_back(char(0xE0 | (c >> 12))); o.push_back(char(0x80 | ((c >> 6) & 63))); o.push_back(char(0x80 | (c & 63))); }
    else { o.push_back(char(0xF0 | (c >> 18))); o.push_back(char(0x80 | ((c >> 12) & 63))); o.push_back(char(0x80 | ((c >> 6) & 63))); o.push_back(char(0x80 | (c & 63))); }
  }
  bool Hex4(std::uint32_t* v) {
    if (i + 4 > s.size()) return Fail("truncated \\u escape");
    std::uint32_t r = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i + k]; r <<= 4;
      if (c >= '0' && c <= '9') r |= c - '0';
      else if (c >= 'a' && c <= 'f') r |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') r |= c - 'A' + 10;
      else return Fail("bad \\u escape");
    }
    i += 4; *v = r; return true;
  }
  bool Str(std::string* o) {
    ++i;  // opening quote
    for (;;) {
      if (i >= s.size()) return Fail("unterminated string");
      unsigned char c = static_cast<unsigned char>(s[i]);
      if (c == '"') { ++i; return true; }
      if (c < 0x20) return Fail("control character in string");
      if (c != '\\') { o->push_back(char(c)); ++i; continue; }
      if (++i >= s.size()) return Fail("bad escape");
      char e = s[i++];
      switch (e) {
        case '"': o->push_back('"'); break;
        case '\\': o->push_back('\\'); break;
        case '/': o->push_back('/'); break;
        case 'b': o->push_back('\b'); break;
        case 'f': o->push_back('\f'); break;
        case 'n': o->push_back('\n'); break;
        case 'r': o->push_back('\r'); break;
        case 't': o->push_back('\t'); break;
        case 'u': {
          std::uint32_t u;
          if (!Hex4(&u)) return false;
          if (u >= 0xD800 && u < 0xDC00) {
            std::uint32_t lo;
            if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
              std::size_t save = i; i += 2;
              if (!Hex4(&lo)) return false;
              if (lo >= 0xDC00 && lo < 0xE000) { PutUtf8(*o, 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00)); break; }
              i = save;
            }
            PutUtf8(*o, 0xFFFD);
          } else if (u >= 0xDC00 && u < 0xE000) {
            PutUtf8(*o, 0xFFFD);
          } else {
            PutUtf8(*o, u);
          }
          break;
        }
        default: return Fail("bad escape");
      }
    }
  }
  bool Lit(std::string_view w) {
    if (s.substr(i, w.size()) == w) { i += w.size(); return true; }
    return false;
  }
  bool Num(JsonValue* out) {
    std::size_t b = i;
    if (i < s.size() && s[i] == '-') ++i;
    if (i < s.size() && s[i] == '0') ++i;
    else if (i < s.size() && s[i] >= '1' && s[i] <= '9') { while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i; }
    else return Fail("bad number");
    bool fl = false;
    if (i + 1 < s.size() && s[i] == '.' && s[i + 1] >= '0' && s[i + 1] <= '9') {
      fl = true; ++i; while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      std::size_t k = i + 1;
      if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
      if (k < s.size() && s[k] >= '0' && s[k] <= '9') {
        while (k < s.size() && s[k] >= '0' && s[k] <= '9') ++k;
        i = k; fl = true;
      }
    }
    std::string_view t = s.substr(b, i - b);
    if (!fl) {
      std::string d(t);
      if (d == "-0") d = "0";
      *out = JsonValue::BigInt(std::move(d));
      return true;
    }
    double v = 0;
    auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    if (r.ec == std::errc::result_out_of_range) {
      // overflow -> inf, underflow -> 0 (as Python's float())
      std::string tmp(t);
      v = std::strtod(tmp.c_str(), nullptr);
    } else if (r.ec != std::errc()) {
      return Fail("bad number");
    }
    *out = JsonValue::Double(v);
    return true;
  }
  bool Value(JsonValue* out) {
    Ws();
    if (i >= s.size()) return Fail("unexpected end");
    char c = s[i];
    if (++depth > 512) return Fail("too deep");
    bool ok = ValueInner(c, out);
    --depth;
    return ok;
  }
  bool ValueInner(char c, JsonValue* out) {
    if (c == '{') {
      ++i; JsonValue::Object obj; Ws();
      if (i < s.size() && s[i] == '}') { ++i; *out = JsonValue::MakeObject(std::move(obj)); return true; }
      for (;;) {
        Ws();
        if (i >= s.size() || s[i] != '"') return Fail("expected key");
        std::string k; if (!Str(&k)) return false;
        Ws(); if (i >= s.size() || s[i] != ':') return Fail("expected ':'"); ++i;
        JsonValue v; if (!Value(&v)) return false;
        bool dup = false;
        for (auto& m : obj) if (m.first == k) { m.second = std::move(v); dup = true; break; }
        if (!dup) obj.emplace_back(std::move(k), std::move(v));
        Ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; break; }
        return Fail("expected ',' or '}'");
      }
      *out = JsonValue::MakeObject(std::move(obj)); return true;
    }
    if (c == '[') {
      ++i; JsonValue::Array arr; Ws();
      if (i < s.size() && s[i] == ']') { ++i; *out = JsonValue::MakeArray(std::move(arr)); return true; }
      for (;;) {
        JsonValue v; if (!Value(&v)) return false;
        arr.push_back(std::move(v)); Ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == ']') { ++i; break; }
        return Fail("expected ',' or ']'");
      }
      *out = JsonValue::MakeArray(std::move(arr)); return true;
    }
    if (c == '"') { std::string v; if (!Str(&v)) return false; *out = JsonValue::String(std::move(v)); return true; }
    if (Lit("true")) { *out = JsonValue::Bool(true); return true; }
    if (Lit("false")) { *out = JsonValue::Bool(false); return true; }
    if (Lit("null")) { *out = JsonValue(); return true; }
    if (Lit("NaN")) { *out = JsonValue::Double(std::nan("")); return true; }
    if (Lit("Infinity")) { *out = JsonValue::Double(HUGE_VAL); return true; }
    if (Lit("-Infinity")) { *out = JsonValue::Double(-HUGE_VAL); return true; }
    return Num(out);
  }
};

// Python repr(float): shortest round-trip digits, exponent form iff exp10 < -4
// or >= 16, ".0" appended to integral fixed form, exponent has sign and >=2 digits.
std::string PyFloatRepr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string sci(buf, r.ptr);  // [-]d[.ddd]e[+-]XX
  bool neg = sci[0] == '-';
  if (neg) sci.erase(0, 1);
  std::size_t epos = sci.find('e');
  int e10 = std::atoi(sci.c_str() + epos + 1);
  std::string digits;
  for (std::size_t k = 0; k < epos; ++k) if (sci[k] != '.') digits.push_back(sci[k]);
  std::string out = neg ? "-" : "";
  if (e10 < -4 || e10 >= 16) {
    out += digits.substr(0, 1);
    if (digits.size() > 1) { out += '.'; out += digits.substr(1); }
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", e10 < 0 ? '-' : '+', e10 < 0 ? -e10 : e10);
    out += eb;
  } else if (e10 < 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-e10 - 1), '0');
    out += digits;
  } else if (static_cast<std::size_t>(e10) + 1 >= digits.size()) {
    out += digits;
    out.append(static_cast<std::size_t>(e10) + 1 - digits.size(), '0');
    out += ".0";
  } else {
    out += digits.substr(0, static_cast<std::size_t>(e10) + 1);
    out += '.';
    out += digits.substr(static_cast<std::size_t>(e10) + 1);
  }
  return out;
}

void AppendJsonString(std::string& o, const std::string& s) {
  o.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o.push_back(char(c));
    }
  }
  o.push_back('"');
}

}  // namespace

bool ParseJson(std::string_view text, JsonValue* out, std::string* error) {
  Parser p;
  p.s = text;
  JsonValue v;
  bool ok = p.Value(&v);
  if (ok) { p.Ws(); if (p.i != text.size()) ok = p.Fail("trailing data"); }
  if (!ok) { if (error) *error = p.err; return false; }
  *out = std::move(v);
  return true;
}

void AppendPythonJson(std::string& o, const JsonValue& v) {
  switch (v.type()) {
    case JsonValue::Type::kNull: o += "null"; break;
    case JsonValue::Type::kBool: o += v.bool_value() ? "true" : "false"; break;
    case JsonValue::Type::kInt: o += v.text(); break;
    case JsonValue::Type::kDouble: o += PyFloatRepr(v.double_value()); break;
    case JsonValue::Type::kString: AppendJsonString(o, v.text()); break;
    case JsonValue::Type::kArray: {
      o.push_back('[');
      bool first = true;
      for (const auto& e : v.array()) { if (!first) o += ", "; first = false; AppendPythonJson(o, e); }
      o.push_back(']');
      break;
    }
    case JsonValue::Type::kObject: {
      o.push_back('{');
      bool first = true;
      for (const auto& m : v.object()) {
        if (!first) o += ", ";
        first = false;
        AppendJsonString(o, m.first); o += ": "; AppendPythonJson(o, m.second);
      }
      o.push_back('}');
      break;
    }
  }
}

std::string DumpPythonJson(const JsonValue& v) {
  std::string o; AppendPythonJson(o, v); return o;
}

}  // namespace ricursive::text
