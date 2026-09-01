/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

constexpr auto pi = 3.14159265358979323846f;

class IESTextParser final {
private:
  std::string _text;
  char *_data{};
  bool _error{};

public:
  explicit IESTextParser(std::string_view text) : _text{text} {
    std::replace(_text.begin(), _text.end(), ',', ' ');
    _data = std::strstr(_text.data(), "\nTILT=");
  }

  [[nodiscard]] bool eof() const noexcept {
    return _data == nullptr || *_data == '\0';
  }
  [[nodiscard]] bool has_error() const noexcept { return _error; }
  [[nodiscard]] char *data() const noexcept { return _data; }
  void set_data(char *data) noexcept { _data = data; }

  double get_double() noexcept {
    if (eof()) {
      _error = true;
      return 0.0;
    }
    auto *old_data = _data;
    const auto value = std::strtod(_data, &_data);
    if (_data == old_data) {
      _data = nullptr;
      _error = true;
      return 0.0;
    }
    return value;
  }

  long get_long() noexcept {
    if (eof()) {
      _error = true;
      return 0;
    }
    auto *old_data = _data;
    const auto value = std::strtol(_data, &_data, 10);
    if (_data == old_data) {
      _data = nullptr;
      _error = true;
      return 0;
    }
    return value;
  }
};

class IESFile final {
private:
  enum class Type : long { c = 1, b = 2, a = 3 };

  Type _type{Type::c};
  std::vector<float> _vertical_angles;
  std::vector<float> _horizontal_angles;
  std::vector<std::vector<float>> _intensity;

  [[nodiscard]] static bool angle_close(float lhs, float rhs) noexcept {
    return std::abs(lhs - rhs) < 1.0e-4f;
  }

  [[nodiscard]] bool parse(std::string_view ies) {
    if (ies.empty()) {
      return false;
    }

    IESTextParser parser{ies};
    if (parser.eof()) {
      return false;
    }

    if (std::strncmp(parser.data(), "\nTILT=INCLUDE", 13u) == 0) {
      parser.set_data(parser.data() + 13u);
      static_cast<void>(parser.get_double());
      const auto tilt_count = parser.get_long();
      if (tilt_count < 0 ||
          static_cast<std::size_t>(tilt_count) > ies.size()) {
        return false;
      }
      // Keep Cycles' "angles followed by factors" consumption order without
      // forming 2 * tilt_count in the signed `long` domain.
      for (auto sequence = 0; sequence < 2; ++sequence) {
        for (auto index = 0l; index < tilt_count; ++index) {
          static_cast<void>(parser.get_double());
        }
      }
    } else {
      parser.set_data(std::strstr(parser.data() + 1u, "\n"));
    }

    if (parser.eof()) {
      return false;
    }
    parser.set_data(parser.data() + 1u);

    static_cast<void>(parser.get_long());
    static_cast<void>(parser.get_double());
    auto factor = parser.get_double();
    const auto vertical_count = parser.get_long();
    const auto horizontal_count = parser.get_long();
    const auto type = parser.get_long();
    if (type != static_cast<long>(Type::a) &&
        type != static_cast<long>(Type::b) &&
        type != static_cast<long>(Type::c)) {
      return false;
    }
    _type = static_cast<Type>(type);

    static_cast<void>(parser.get_long());
    static_cast<void>(parser.get_double());
    static_cast<void>(parser.get_double());
    static_cast<void>(parser.get_double());
    factor *= parser.get_double();
    factor *= parser.get_double();
    static_cast<void>(parser.get_double());

    // Cycles stores counts as int and later indexes h * v. Reject only the
    // malformed region in which its implementation can overflow or reserve
    // more numeric entries than the complete source could contain.
    if (vertical_count <= 0 || horizontal_count <= 0 ||
        vertical_count > std::numeric_limits<int>::max() ||
        horizontal_count > std::numeric_limits<int>::max()) {
      return false;
    }
    const auto v_count = static_cast<std::size_t>(vertical_count);
    const auto h_count = static_cast<std::size_t>(horizontal_count);
    if (v_count > ies.size() || h_count > ies.size() ||
        h_count > ies.size() / v_count) {
      return false;
    }

    // Cycles 5.2.1's fixed D65 luminous-efficacy conversion, including the
    // 4*pi factor expected by lamp strength.
    factor *= 0.0706650768394;

    _vertical_angles.reserve(v_count);
    for (auto index = std::size_t{}; index < v_count; ++index) {
      _vertical_angles.emplace_back(
          static_cast<float>(parser.get_double()));
    }
    _horizontal_angles.reserve(h_count);
    for (auto index = std::size_t{}; index < h_count; ++index) {
      _horizontal_angles.emplace_back(
          static_cast<float>(parser.get_double()));
    }
    _intensity.resize(h_count);
    for (auto h = std::size_t{}; h < h_count; ++h) {
      _intensity[h].reserve(v_count);
      for (auto v = std::size_t{}; v < v_count; ++v) {
        _intensity[h].emplace_back(
            static_cast<float>(factor * parser.get_double()));
      }
    }
    return !parser.has_error();
  }

  void process_type_b() {
    std::vector<std::vector<float>> transposed(_vertical_angles.size());
    for (auto v = std::size_t{}; v < _vertical_angles.size(); ++v) {
      transposed[v].reserve(_horizontal_angles.size());
      for (auto h = std::size_t{}; h < _horizontal_angles.size(); ++h) {
        transposed[v].emplace_back(_intensity[h][v]);
      }
    }
    _intensity.swap(transposed);
    _horizontal_angles.swap(_vertical_angles);

    if (angle_close(_horizontal_angles.front(), 0.0f)) {
      const auto count = _horizontal_angles.size();
      std::vector<float> angles;
      std::vector<std::vector<float>> values;
      angles.reserve(2u * count - 1u);
      values.reserve(2u * count - 1u);
      for (auto index = count; index-- > 1u;) {
        angles.emplace_back(90.0f - _horizontal_angles[index]);
        values.emplace_back(_intensity[index]);
      }
      for (auto index = std::size_t{}; index < count; ++index) {
        angles.emplace_back(90.0f + _horizontal_angles[index]);
        values.emplace_back(_intensity[index]);
      }
      _horizontal_angles.swap(angles);
      _intensity.swap(values);
    } else {
      for (auto &angle : _horizontal_angles) {
        angle += 90.0f;
      }
    }

    if (angle_close(_vertical_angles.front(), 0.0f)) {
      const auto vertical_count = _vertical_angles.size();
      std::vector<float> angles;
      angles.reserve(2u * vertical_count - 1u);
      for (auto index = vertical_count; index-- > 1u;) {
        angles.emplace_back(90.0f - _vertical_angles[index]);
      }
      for (const auto angle : _vertical_angles) {
        angles.emplace_back(90.0f + angle);
      }
      for (auto &horizontal : _intensity) {
        std::vector<float> values;
        values.reserve(2u * vertical_count - 1u);
        for (auto index = vertical_count; index-- > 1u;) {
          values.emplace_back(horizontal[index]);
        }
        values.insert(values.end(), horizontal.begin(), horizontal.end());
        horizontal.swap(values);
      }
      _vertical_angles.swap(angles);
    } else {
      for (auto &angle : _vertical_angles) {
        angle += 90.0f;
      }
    }
  }

  void process_type_a() {
    for (auto &angle : _vertical_angles) {
      angle += 90.0f;
    }

    std::vector<float> angles;
    std::vector<std::vector<float>> values;
    angles.reserve(2u * _horizontal_angles.size() - 1u);
    values.reserve(2u * _horizontal_angles.size() - 1u);
    for (auto index = _horizontal_angles.size(); index-- > 0u;) {
      angles.emplace_back(180.0f - _horizontal_angles[index]);
      values.emplace_back(_intensity[index]);
    }
    if (angle_close(_horizontal_angles.front(), 0.0f)) {
      for (auto index = std::size_t{1u};
           index < _horizontal_angles.size(); ++index) {
        angles.emplace_back(180.0f + _horizontal_angles[index]);
        values.emplace_back(_intensity[index]);
      }
    }
    _horizontal_angles.swap(angles);
    _intensity.swap(values);
  }

  void process_type_c() {
    if (angle_close(_horizontal_angles.front(), 90.0f)) {
      for (auto &angle : _horizontal_angles) {
        angle -= 90.0f;
      }
    }
    if (_horizontal_angles.size() == 1u) {
      _horizontal_angles.front() = 0.0f;
      _horizontal_angles.emplace_back(360.0f);
      _intensity.emplace_back(_intensity.front());
    }
    if (angle_close(_horizontal_angles.back(), 90.0f)) {
      const auto count = _horizontal_angles.size();
      for (auto index = count - 1u; index-- > 0u;) {
        _horizontal_angles.emplace_back(
            180.0f - _horizontal_angles[index]);
        _intensity.emplace_back(_intensity[index]);
      }
    }
    if (angle_close(_horizontal_angles.back(), 180.0f)) {
      const auto count = _horizontal_angles.size();
      for (auto index = count - 1u; index-- > 0u;) {
        _horizontal_angles.emplace_back(
            360.0f - _horizontal_angles[index]);
        _intensity.emplace_back(_intensity[index]);
      }
    }
    if (angle_close(_horizontal_angles.front(), 0.0f) &&
        !angle_close(_horizontal_angles.back(), 360.0f) &&
        _horizontal_angles.size() >= 2u) {
      const auto count = _horizontal_angles.size();
      const auto last_step = _horizontal_angles[count - 1u] -
                             _horizontal_angles[count - 2u];
      const auto first_step = _horizontal_angles[1u] -
                              _horizontal_angles[0u];
      const auto gap_step = 360.0f - _horizontal_angles.back();
      if (angle_close(last_step, gap_step) ||
          angle_close(first_step, gap_step)) {
        _horizontal_angles.emplace_back(360.0f);
        _intensity.emplace_back(_intensity.front());
      }
    }
  }

  [[nodiscard]] bool process() {
    if (_horizontal_angles.empty() || _vertical_angles.empty()) {
      return false;
    }
    switch (_type) {
      case Type::a:
        process_type_a();
        break;
      case Type::b:
        process_type_b();
        break;
      case Type::c:
        process_type_c();
        break;
    }
    // The device interpolation reads angle i+1 in both dimensions. Cycles'
    // behavior below two samples is undefined, so malformed data becomes the
    // ordinary invalid-profile slot instead of an out-of-bounds GPU read.
    if (_horizontal_angles.size() < 2u || _vertical_angles.size() < 2u) {
      return false;
    }
    if (_horizontal_angles.size() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        _vertical_angles.size() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return false;
    }
    const auto strictly_increasing = [](const std::vector<float> &angles) {
      return std::all_of(angles.begin(), angles.end(),
                         [](float angle) { return std::isfinite(angle); }) &&
             std::adjacent_find(
                 angles.begin(), angles.end(),
                 [](float lhs, float rhs) { return !(lhs < rhs); }) ==
             angles.end();
    };
    if (!strictly_increasing(_horizontal_angles) ||
        !strictly_increasing(_vertical_angles)) {
      return false;
    }
    for (auto &angle : _vertical_angles) {
      angle *= pi / 180.0f;
    }
    for (auto &angle : _horizontal_angles) {
      angle *= pi / 180.0f;
    }
    return true;
  }

public:
  [[nodiscard]] bool load(std::string_view ies) {
    _vertical_angles.clear();
    _horizontal_angles.clear();
    _intensity.clear();
    if (!parse(ies) || !process()) {
      _vertical_angles.clear();
      _horizontal_angles.clear();
      _intensity.clear();
      return false;
    }
    return true;
  }

  [[nodiscard]] std::vector<float> pack() const {
    if (_horizontal_angles.empty() || _vertical_angles.empty()) {
      return {};
    }
    std::vector<float> data;
    data.reserve(2u + _horizontal_angles.size() + _vertical_angles.size() +
                 _horizontal_angles.size() * _vertical_angles.size());
    data.emplace_back(std::bit_cast<float>(
        static_cast<std::int32_t>(_horizontal_angles.size())));
    data.emplace_back(std::bit_cast<float>(
        static_cast<std::int32_t>(_vertical_angles.size())));
    data.insert(data.end(), _horizontal_angles.begin(),
                _horizontal_angles.end());
    data.insert(data.end(), _vertical_angles.begin(), _vertical_angles.end());
    for (const auto &values : _intensity) {
      data.insert(data.end(), values.begin(), values.end());
    }
    return data;
  }
};

} // namespace

std::uint32_t IESIDMap::get_ies_slot(std::string_view content) {
  const std::scoped_lock lock{_ies_lock};
  const auto key = std::string{content};
  if (const auto iter = _ies_slots.find(key); iter != _ies_slots.end()) {
    return iter->second;
  }
  if (_packed_profiles.size() >=
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::length_error{"Cycles IES slot table exceeds int32 offsets"};
  }
  IESFile profile;
  static_cast<void>(profile.load(content));
  const auto slot = static_cast<std::uint32_t>(_packed_profiles.size());
  _packed_profiles.emplace_back(profile.pack());
  _ies_slots.emplace(key, slot);
  return slot;
}

std::vector<float> IESIDMap::packed_data() const {
  const std::scoped_lock lock{_ies_lock};
  auto size = _packed_profiles.size();
  if (size >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::length_error{"Cycles IES slot table exceeds int32 offsets"};
  }
  for (const auto &profile : _packed_profiles) {
    if (profile.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) -
            size) {
      throw std::length_error{"Cycles IES packed table exceeds int32 offsets"};
    }
    size += profile.size();
  }

  std::vector<float> data(size);
  auto offset = _packed_profiles.size();
  for (auto slot = std::size_t{}; slot < _packed_profiles.size(); ++slot) {
    const auto &profile = _packed_profiles[slot];
    if (profile.empty()) {
      data[slot] = std::bit_cast<float>(std::int32_t{-1});
      continue;
    }
    data[slot] =
        std::bit_cast<float>(static_cast<std::int32_t>(offset));
    std::copy(profile.begin(), profile.end(),
              data.begin() + static_cast<std::ptrdiff_t>(offset));
    offset += profile.size();
  }
  return data;
}

std::size_t IESIDMap::slot_count() const {
  const std::scoped_lock lock{_ies_lock};
  return _packed_profiles.size();
}

} // namespace psycles::compiler::cycles_svm
