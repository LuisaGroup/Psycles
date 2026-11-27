#include <psycles/luisa/cycles_spectrum.h>

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace psycles::luisa_backend::cycles_spectrum {
namespace {

struct Rgb {
  double x;
  double y;
  double z;
};

// Standard CIE 1931 2-deg color matching functions at 5 nm spacing from
// 360 nm to 830 nm (95 entries: 360..375 nm tail, 380..780 nm matching
// cycles_color_nodes::cie_color_match, and 785..830 nm NIR tail).
constexpr std::array<Rgb, 95u> cie_1931_5nm{{
    // 360, 365, 370, 375 nm
    {0.0001299, 0.000003917, 0.0006061},
    {0.0002321, 0.000006965, 0.001086},
    {0.0004149, 0.00001239, 0.001946},
    {0.0007416, 0.00002202, 0.003486},
    // 380..780 nm (81 entries matching cycles_color_nodes::cie_color_match)
    {0.0014, 0.0000, 0.0065},
    {0.0022, 0.0001, 0.0105},
    {0.0042, 0.0001, 0.0201},
    {0.0076, 0.0002, 0.0362},
    {0.0143, 0.0004, 0.0679},
    {0.0232, 0.0006, 0.1102},
    {0.0435, 0.0012, 0.2074},
    {0.0776, 0.0022, 0.3713},
    {0.1344, 0.0040, 0.6456},
    {0.2148, 0.0073, 1.0391},
    {0.2839, 0.0116, 1.3856},
    {0.3285, 0.0168, 1.6230},
    {0.3483, 0.0230, 1.7471},
    {0.3481, 0.0298, 1.7826},
    {0.3362, 0.0380, 1.7721},
    {0.3187, 0.0480, 1.7441},
    {0.2908, 0.0600, 1.6692},
    {0.2511, 0.0739, 1.5281},
    {0.1954, 0.0910, 1.2876},
    {0.1421, 0.1126, 1.0419},
    {0.0956, 0.1390, 0.8130},
    {0.0580, 0.1693, 0.6162},
    {0.0320, 0.2080, 0.4652},
    {0.0147, 0.2586, 0.3533},
    {0.0049, 0.3230, 0.2720},
    {0.0024, 0.4073, 0.2123},
    {0.0093, 0.5030, 0.1582},
    {0.0291, 0.6082, 0.1117},
    {0.0633, 0.7100, 0.0782},
    {0.1096, 0.7932, 0.0573},
    {0.1655, 0.8620, 0.0422},
    {0.2257, 0.9149, 0.0298},
    {0.2904, 0.9540, 0.0203},
    {0.3597, 0.9803, 0.0134},
    {0.4334, 0.9950, 0.0087},
    {0.5121, 1.0000, 0.0057},
    {0.5945, 0.9950, 0.0039},
    {0.6784, 0.9786, 0.0027},
    {0.7621, 0.9520, 0.0021},
    {0.8425, 0.9154, 0.0018},
    {0.9163, 0.8700, 0.0017},
    {0.9786, 0.8163, 0.0014},
    {1.0263, 0.7570, 0.0011},
    {1.0567, 0.6949, 0.0010},
    {1.0622, 0.6310, 0.0008},
    {1.0456, 0.5668, 0.0006},
    {1.0026, 0.5030, 0.0003},
    {0.9384, 0.4412, 0.0002},
    {0.8544, 0.3810, 0.0002},
    {0.7514, 0.3210, 0.0001},
    {0.6424, 0.2650, 0.0000},
    {0.5419, 0.2170, 0.0000},
    {0.4479, 0.1750, 0.0000},
    {0.3608, 0.1382, 0.0000},
    {0.2835, 0.1070, 0.0000},
    {0.2187, 0.0816, 0.0000},
    {0.1649, 0.0610, 0.0000},
    {0.1212, 0.0446, 0.0000},
    {0.0874, 0.0320, 0.0000},
    {0.0636, 0.0232, 0.0000},
    {0.0468, 0.0170, 0.0000},
    {0.0329, 0.0119, 0.0000},
    {0.0227, 0.0082, 0.0000},
    {0.0158, 0.0057, 0.0000},
    {0.0114, 0.0041, 0.0000},
    {0.0081, 0.0029, 0.0000},
    {0.0058, 0.0021, 0.0000},
    {0.0041, 0.0015, 0.0000},
    {0.0029, 0.0010, 0.0000},
    {0.0020, 0.0007, 0.0000},
    {0.0014, 0.0005, 0.0000},
    {0.0010, 0.0004, 0.0000},
    {0.0007, 0.0002, 0.0000},
    {0.0005, 0.0002, 0.0000},
    {0.0003, 0.0001, 0.0000},
    {0.0002, 0.0001, 0.0000},
    {0.0002, 0.0001, 0.0000},
    {0.0001, 0.0000, 0.0000},
    {0.0001, 0.0000, 0.0000},
    {0.0001, 0.0000, 0.0000},
    {0.0000415, 0.0000150, 0.0},
    // 785..830 nm
    {0.0000291, 0.0000105, 0.0},
    {0.0000204, 0.0000074, 0.0},
    {0.0000143, 0.0000052, 0.0},
    {0.0000100, 0.0000036, 0.0},
    {0.0000071, 0.0000025, 0.0},
    {0.0000050, 0.0000018, 0.0},
    {0.0000035, 0.0000013, 0.0},
    {0.0000025, 0.0000009, 0.0},
    {0.0000017, 0.0000006, 0.0},
    {0.0000012, 0.0000004, 0.0},
}};

// Standard CIE 15:2004 / ISO 11664-2 Illuminant D65 SPD at 10 nm (360..830 nm).
constexpr std::array<double, 48u> cie_d65_10nm{{
    46.6383, 52.0891, 49.9755, 54.6482, 82.7549, 91.4860, 93.4318, 86.6823,
    104.865, 117.008, 117.812, 114.861, 115.923, 108.811, 109.354, 107.802,
    104.790, 107.689, 104.405, 104.046, 100.000, 96.3342, 95.7880, 88.6856,
    90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268, 80.2146, 82.2778,
    78.2842, 69.7213, 71.6091, 74.3490, 61.6040, 69.8856, 75.0870, 63.5927,
    46.4182, 66.8054, 63.3828, 64.3040, 59.4519, 51.9590, 57.4406, 60.3125,
}};

[[nodiscard]] double smoothstep(double x) noexcept {
  const double clamped = std::clamp(x, 0.0, 1.0);
  return clamped * clamped * (3.0 - 2.0 * clamped);
}

[[nodiscard]] double inv_smoothstep(double v) noexcept {
  const double arg = std::clamp(1.0 - 2.0 * v, -1.0, 1.0);
  return 0.5 - std::sin(std::asin(arg) / 3.0);
}

[[nodiscard]] std::uint64_t hash_color_space(
    const contract::ShaderColorSpace &cs) noexcept {
  std::uint64_t h = 14695981039346656037ull;
  auto mix_float = [&](float v) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    for (int i = 0; i < 4; ++i) {
      h ^= static_cast<std::uint8_t>((bits >> (i * 8)) & 0xffu);
      h *= 1099511628211ull;
    }
  };
  auto mix_vec = [&](const Vec3f &v) noexcept {
    mix_float(v.x);
    mix_float(v.y);
    mix_float(v.z);
  };
  mix_vec(cs.xyz_to_r);
  mix_vec(cs.xyz_to_g);
  mix_vec(cs.xyz_to_b);
  return h;
}

[[nodiscard]] std::array<std::array<double, 3u>, 3u> invert_3x3(
    const contract::ShaderColorSpace &cs) noexcept {
  const double m[3][3] = {
      {cs.xyz_to_r.x, cs.xyz_to_r.y, cs.xyz_to_r.z},
      {cs.xyz_to_g.x, cs.xyz_to_g.y, cs.xyz_to_g.z},
      {cs.xyz_to_b.x, cs.xyz_to_b.y, cs.xyz_to_b.z},
  };
  const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const double c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
  const double c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
  const double c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
  const double c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
  const double c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
  const double c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  const double inv_det = std::abs(det) > 1.0e-20 ? 1.0 / det : 1.0;
  return {{
      {c00 * inv_det, c10 * inv_det, c20 * inv_det},
      {c01 * inv_det, c11 * inv_det, c21 * inv_det},
      {c02 * inv_det, c12 * inv_det, c22 * inv_det},
  }};
}

[[nodiscard]] bool solve_linear_3x3(const double a_in[3][3],
                                    const double b_in[3],
                                    double x_out[3]) noexcept {
  double a[3][3]{};
  double b[3]{};
  for (int i = 0; i < 3; ++i) {
    b[i] = b_in[i];
    for (int j = 0; j < 3; ++j) {
      a[i][j] = a_in[i][j];
    }
  }
  for (int col = 0; col < 3; ++col) {
    int pivot = col;
    double best = std::abs(a[col][col]);
    for (int row = col + 1; row < 3; ++row) {
      if (std::abs(a[row][col]) > best) {
        best = std::abs(a[row][col]);
        pivot = row;
      }
    }
    if (best < 1.0e-18) {
      return false;
    }
    if (pivot != col) {
      for (int k = col; k < 3; ++k) {
        std::swap(a[col][k], a[pivot][k]);
      }
      std::swap(b[col], b[pivot]);
    }
    const double inv_diag = 1.0 / a[col][col];
    for (int row = col + 1; row < 3; ++row) {
      const double f = a[row][col] * inv_diag;
      for (int k = col + 1; k < 3; ++k) {
        a[row][k] -= f * a[col][k];
      }
      b[row] -= f * b[col];
    }
  }
  for (int row = 2; row >= 0; --row) {
    double sum = b[row];
    for (int k = row + 1; k < 3; ++k) {
      sum -= a[row][k] * x_out[k];
    }
    x_out[row] = sum / a[row][row];
  }
  return true;
}

struct QuadratureWorkspace {
  std::array<double, spectral_cie_sample_count> t{};
  std::array<double, spectral_cie_sample_count> wx{};
  std::array<double, spectral_cie_sample_count> wy{};
  std::array<double, spectral_cie_sample_count> wz{};
  std::array<std::array<double, 3u>, 3u> rgb_to_xyz{};
};

void solve_sigmoid_poly(const QuadratureWorkspace &ws,
                        const std::array<double, 3u> &rgb,
                        std::array<double, 3u> &c) noexcept {
  const double target[3] = {
      ws.rgb_to_xyz[0][0] * rgb[0] + ws.rgb_to_xyz[0][1] * rgb[1] +
          ws.rgb_to_xyz[0][2] * rgb[2],
      ws.rgb_to_xyz[1][0] * rgb[0] + ws.rgb_to_xyz[1][1] * rgb[1] +
          ws.rgb_to_xyz[1][2] * rgb[2],
      ws.rgb_to_xyz[2][0] * rgb[0] + ws.rgb_to_xyz[2][1] * rgb[1] +
          ws.rgb_to_xyz[2][2] * rgb[2],
  };

  auto eval_res = [&](const std::array<double, 3u> &coeffs, double res[3],
                      double *jac_flat) noexcept -> double {
    double xyz[3]{};
    double jtj[3][3]{};
    double j[3][3]{};
    for (std::size_t idx = 0u; idx < spectral_cie_sample_count; ++idx) {
      const double ti = ws.t[idx];
      const double p = (coeffs[0] * ti + coeffs[1]) * ti + coeffs[2];
      const double denom = std::sqrt(1.0 + p * p);
      const double s = 0.5 + 0.5 * p / denom;
      xyz[0] += s * ws.wx[idx];
      xyz[1] += s * ws.wy[idx];
      xyz[2] += s * ws.wz[idx];
      if (jac_flat != nullptr) {
        const double ds = 0.5 / (denom * denom * denom);
        const double t2 = ti * ti;
        const double w[3] = {ws.wx[idx], ws.wy[idx], ws.wz[idx]};
        for (int a = 0; a < 3; ++a) {
          const double wa = w[a] * ds;
          j[a][0] += wa * t2;
          j[a][1] += wa * ti;
          j[a][2] += wa;
        }
      }
    }
    res[0] = xyz[0] - target[0];
    res[1] = xyz[1] - target[1];
    res[2] = xyz[2] - target[2];
    if (jac_flat != nullptr) {
      (void)jtj;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          jac_flat[a * 3 + b] = j[a][b];
        }
      }
    }
    return res[0] * res[0] + res[1] * res[1] + res[2] * res[2];
  };

  for (int iter = 0; iter < 40; ++iter) {
    double r[3]{};
    double jac[9]{};
    const double err2 = eval_res(c, r, jac);
    if (err2 < 1.0e-18) {
      break;
    }
    double grad[3]{};
    double h[3][3]{};
    for (int col = 0; col < 3; ++col) {
      grad[col] = jac[0 * 3 + col] * r[0] + jac[1 * 3 + col] * r[1] +
                  jac[2 * 3 + col] * r[2];
      for (int col2 = 0; col2 < 3; ++col2) {
        h[col][col2] = jac[0 * 3 + col] * jac[0 * 3 + col2] +
                       jac[1 * 3 + col] * jac[1 * 3 + col2] +
                       jac[2 * 3 + col] * jac[2 * 3 + col2];
      }
    }
    double lambda = 1.0e-10;
    bool accepted = false;
    for (int attempt = 0; attempt < 12; ++attempt) {
      double h_damped[3][3]{};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          h_damped[i][j] = h[i][j] + (i == j ? lambda : 0.0);
        }
      }
      const double neg_grad[3] = {-grad[0], -grad[1], -grad[2]};
      double dc[3]{};
      if (solve_linear_3x3(h_damped, neg_grad, dc)) {
        std::array<double, 3u> cn{
            std::clamp(c[0] + std::clamp(dc[0], -25.0, 25.0), -1000.0, 1000.0),
            std::clamp(c[1] + std::clamp(dc[1], -25.0, 25.0), -1000.0, 1000.0),
            std::clamp(c[2] + std::clamp(dc[2], -25.0, 25.0), -1000.0, 1000.0),
        };
        double rn[3]{};
        const double err2_n = eval_res(cn, rn, nullptr);
        if (err2_n <= err2) {
          c = cn;
          accepted = true;
          break;
        }
      }
      lambda *= 10.0;
    }
    if (!accepted) {
      break;
    }
  }
}

struct DiskCacheHeader {
  std::uint64_t magic{0x5053595350454332ull};
  std::uint32_t res{rgb2spec_table_resolution};
  std::uint32_t cie_count{spectral_cie_sample_count};
  std::uint32_t bb_count{blackbody_lut_count};
  std::uint32_t reserved{0u};
  std::uint64_t cs_hash{};
};

[[nodiscard]] std::filesystem::path cache_file_path(std::uint64_t cs_hash) {
  std::error_code ec;
  auto dir = std::filesystem::temp_directory_path(ec);
  if (ec || dir.empty()) {
    return {};
  }
  dir /= "psycles_spectral_lut";
  char hex[32]{};
  std::snprintf(hex, sizeof(hex), "lut_v2_r%u_%016llx.bin",
                rgb2spec_table_resolution,
                static_cast<unsigned long long>(cs_hash));
  return dir / hex;
}

[[nodiscard]] bool try_load_cached(const std::filesystem::path &path,
                                    std::uint64_t cs_hash,
                                    SpectralTablesHost &out) {
  if (path.empty()) {
    return false;
  }
  std::ifstream in{path, std::ios::binary};
  if (!in.is_open()) {
    return false;
  }
  DiskCacheHeader hdr{};
  in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
  if (!in || hdr.magic != DiskCacheHeader{}.magic ||
      hdr.res != rgb2spec_table_resolution ||
      hdr.cie_count != spectral_cie_sample_count ||
      hdr.bb_count != blackbody_lut_count || hdr.cs_hash != cs_hash) {
    return false;
  }
  constexpr std::size_t lut_size =
      3ull * rgb2spec_table_resolution * rgb2spec_table_resolution *
      rgb2spec_table_resolution;
  out.cie_illuminant.resize(spectral_cie_sample_count);
  out.rgb2spec_coeffs.resize(lut_size);
  out.blackbody_norm.resize(blackbody_lut_count);
  in.read(reinterpret_cast<char *>(&out.spec_xyz_to_r), sizeof(luisa::float3) * 4u);
  in.read(reinterpret_cast<char *>(out.cie_illuminant.data()),
          static_cast<std::streamsize>(out.cie_illuminant.size() *
                                       sizeof(luisa::float4)));
  in.read(reinterpret_cast<char *>(out.rgb2spec_coeffs.data()),
          static_cast<std::streamsize>(out.rgb2spec_coeffs.size() *
                                       sizeof(luisa::float4)));
  in.read(reinterpret_cast<char *>(out.blackbody_norm.data()),
          static_cast<std::streamsize>(out.blackbody_norm.size() *
                                       sizeof(float)));
  return static_cast<bool>(in);
}

void try_save_cached(const std::filesystem::path &path, std::uint64_t cs_hash,
                     const SpectralTablesHost &tables) {
  if (path.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    return;
  }
  auto tmp = path;
#ifdef _WIN32
  const auto pid = static_cast<unsigned long>(_getpid());
#else
  const auto pid = static_cast<unsigned long>(getpid());
#endif
  tmp += ".tmp." + std::to_string(pid);
  {
    std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
    if (!out.is_open()) {
      return;
    }
    DiskCacheHeader hdr{};
    hdr.cs_hash = cs_hash;
    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char *>(&tables.spec_xyz_to_r),
              sizeof(luisa::float3) * 4u);
    out.write(reinterpret_cast<const char *>(tables.cie_illuminant.data()),
              static_cast<std::streamsize>(tables.cie_illuminant.size() *
                                           sizeof(luisa::float4)));
    out.write(reinterpret_cast<const char *>(tables.rgb2spec_coeffs.data()),
              static_cast<std::streamsize>(tables.rgb2spec_coeffs.size() *
                                           sizeof(luisa::float4)));
    out.write(reinterpret_cast<const char *>(tables.blackbody_norm.data()),
              static_cast<std::streamsize>(tables.blackbody_norm.size() *
                                           sizeof(float)));
    if (!out) {
      std::filesystem::remove(tmp, ec);
      return;
    }
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
  }
}

[[nodiscard]] double planck_relative_peak_normalized(double lambda_nm,
                                                   double temp_k) noexcept {
  // First/second radiation constant c2 = h*c/k_B in nm * K.
  constexpr double c2 = 1.438776877e7;
  const double peak_lambda = std::clamp(2.897771955e6 / temp_k, 360.0, 830.0);
  const double x_peak = c2 / (peak_lambda * temp_k);
  const double denom_peak =
      x_peak > 1.0e-6 ? 1.0 - std::exp(-x_peak) : x_peak * (1.0 - 0.5 * x_peak);
  const double x = c2 / (lambda_nm * temp_k);
  const double denom =
      x > 1.0e-6 ? 1.0 - std::exp(-x) : x * (1.0 - 0.5 * x);
  const double l_norm = lambda_nm / peak_lambda;
  const double l5 = l_norm * l_norm * l_norm * l_norm * l_norm;
  const double exp_ratio = std::exp(std::clamp(x_peak - x, -700.0, 80.0));
  return (exp_ratio * denom_peak) / std::max(l5 * denom, 1.0e-30);
}

void fill_xyz3_tables(SpectralTablesHost &tables) noexcept {
  // Exact normalized CIE 1931 2-deg PDFs reconstructed from cie_illuminant
  // (.xyz stores CMF / Y_int) using the same trapezoidal rule on [360..830] nm.
  constexpr std::size_t n_cie = spectral_cie_sample_count;
  std::array<std::array<double, n_cie>, 3u> y_val{};
  std::array<std::array<double, n_cie>, 3u> raw_cdf{};
  std::array<double, 3u> integrals{};
  std::array<std::array<double, n_cie>, 3u> pdf{};
  std::array<std::array<double, n_cie>, 3u> cdf{};
  std::array<std::size_t, 3u> last_active{};

  for (std::size_t c = 0u; c < 3u; ++c) {
    for (std::size_t i = 0u; i < n_cie; ++i) {
      const auto v = tables.cie_illuminant[i];
      const double cmf_c = (c == 0u) ? v.x : ((c == 1u) ? v.y : v.z);
      y_val[c][i] = std::max(cmf_c, 0.0);
    }
    raw_cdf[c][0u] = 0.0;
    for (std::size_t i = 0u; i + 1u < n_cie; ++i) {
      raw_cdf[c][i + 1u] =
          raw_cdf[c][i] + 0.5 * (y_val[c][i] + y_val[c][i + 1u]);
    }
    integrals[c] = std::max(raw_cdf[c][n_cie - 1u], 1.0e-30);
    const double inv_integral = 1.0 / integrals[c];
    for (std::size_t i = 0u; i < n_cie; ++i) {
      pdf[c][i] = y_val[c][i] * inv_integral;
      cdf[c][i] = raw_cdf[c][i] * inv_integral;
    }
    cdf[c][0u] = 0.0;
    cdf[c][n_cie - 1u] = 1.0;
    last_active[c] = 0u;
    for (std::size_t i = 0u; i + 1u < n_cie; ++i) {
      if (cdf[c][i + 1u] > cdf[c][i]) {
        last_active[c] = i;
      }
    }
  }

  // 1. Build 2048-entry exact inverse CDF table by inverting piecewise-linear PDFs.
  tables.xyz3_cdfinv.resize(xyz3_cdfinv_lut_count);
  for (std::uint32_t k = 0u; k < xyz3_cdfinv_lut_count; ++k) {
    const double u = static_cast<double>(k) /
                     static_cast<double>(xyz3_cdfinv_lut_count - 1u);
    std::array<float, 3u> lam{};
    for (std::size_t c = 0u; c < 3u; ++c) {
      const std::size_t end_bin = last_active[c];
      auto it = std::upper_bound(cdf[c].begin() + 1,
                                 cdf[c].begin() + end_bin + 1, u);
      const std::size_t idx =
          static_cast<std::size_t>(std::max<std::ptrdiff_t>(
              0, (it - cdf[c].begin()) - 1));
      const std::size_t i = std::min(idx, end_bin);
      const double p0 = pdf[c][i];
      const double p1 = pdf[c][i + 1u];
      const double bin_mass = std::max(cdf[c][i + 1u] - cdf[c][i], 0.0);
      const double delta = std::clamp(u - cdf[c][i], 0.0, bin_mass);
      const double disc = std::max(p0 * p0 + 2.0 * (p1 - p0) * delta, 0.0);
      const double denom = p0 + std::sqrt(disc);
      const double t =
          denom > 1.0e-30 ? std::clamp((2.0 * delta) / denom, 0.0, 1.0) : 0.0;
      const double wl = static_cast<double>(spectral_lambda_min) +
                        static_cast<double>(i) + t;
      lam[c] = static_cast<float>(
          std::clamp(wl, static_cast<double>(spectral_lambda_min),
                     static_cast<double>(spectral_lambda_max)));
    }
    tables.xyz3_cdfinv[k] =
        luisa::make_float4(lam[0u], lam[1u], lam[2u], 0.0f);
  }

  // 2. Build 512-entry exact single-wavelength collapse weight LUT over [360..830] nm.
  tables.xyz3_collapse.resize(xyz3_collapse_lut_count);
  for (std::uint32_t k = 0u; k < xyz3_collapse_lut_count; ++k) {
    const double t = static_cast<double>(k) /
                     static_cast<double>(xyz3_collapse_lut_count - 1u);
    const double pos = t * static_cast<double>(n_cie - 1u);
    const std::size_t i0 =
        std::min(static_cast<std::size_t>(pos), n_cie - 2u);
    const double frac = pos - static_cast<double>(i0);
    const double px = pdf[0u][i0] * (1.0 - frac) + pdf[0u][i0 + 1u] * frac;
    const double py = pdf[1u][i0] * (1.0 - frac) + pdf[1u][i0 + 1u] * frac;
    const double pz = pdf[2u][i0] * (1.0 - frac) + pdf[2u][i0 + 1u] * frac;
    const double sum_p = px + py + pz;
    const double p_mix = sum_p * (1.0 / 3.0);
    if (sum_p > 1.0e-30) {
      const double scale = 3.0 / sum_p;
      tables.xyz3_collapse[k] = luisa::make_float4(
          static_cast<float>(px * scale), static_cast<float>(py * scale),
          static_cast<float>(pz * scale), static_cast<float>(p_mix));
    } else {
      tables.xyz3_collapse[k] = luisa::make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    }
  }

  // 3. Raw xyz3 -> working RGB matrix = M_spec * diag(I_X/I_Y, 1, I_Z/I_Y).
  const double inv_iy = 1.0 / std::max(integrals[1u], 1.0e-30);
  tables.xyz3_integral_ratio = luisa::make_float3(
      static_cast<float>(integrals[0u] * inv_iy), 1.0f,
      static_cast<float>(integrals[2u] * inv_iy));
  tables.xyz3_to_r = tables.spec_xyz_to_r * tables.xyz3_integral_ratio;
  tables.xyz3_to_g = tables.spec_xyz_to_g * tables.xyz3_integral_ratio;
  tables.xyz3_to_b = tables.spec_xyz_to_b * tables.xyz3_integral_ratio;
}

} // namespace

SpectralTablesHost build_spectral_tables_host(
    const contract::ShaderColorSpace &color_space) {
  const std::uint64_t cs_hash = hash_color_space(color_space);
  const auto cache_path = cache_file_path(cs_hash);
  SpectralTablesHost cached;
  if (try_load_cached(cache_path, cs_hash, cached)) {
    fill_xyz3_tables(cached);
    return cached;
  }

  // 1. Interpolate CIE 1931 5nm CMFs and D65 10nm SPD onto 1 nm [360..830].
  std::array<Rgb, spectral_cie_sample_count> cmf_1nm{};
  std::array<double, spectral_cie_sample_count> d65_1nm{};
  std::array<double, spectral_cie_sample_count> w_trap{};
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    const double wl = static_cast<double>(spectral_lambda_min) +
                      static_cast<double>(i);
    const double u5 = (wl - 360.0) / 5.0;
    const std::size_t i5 =
        std::min(static_cast<std::size_t>(u5), cie_1931_5nm.size() - 2u);
    const double f5 = u5 - static_cast<double>(i5);
    cmf_1nm[i] = {
        cie_1931_5nm[i5].x * (1.0 - f5) + cie_1931_5nm[i5 + 1u].x * f5,
        cie_1931_5nm[i5].y * (1.0 - f5) + cie_1931_5nm[i5 + 1u].y * f5,
        cie_1931_5nm[i5].z * (1.0 - f5) + cie_1931_5nm[i5 + 1u].z * f5,
    };

    const double u10 = (wl - 360.0) / 10.0;
    const std::size_t i10 =
        std::min(static_cast<std::size_t>(u10), cie_d65_10nm.size() - 2u);
    const double f10 = u10 - static_cast<double>(i10);
    d65_1nm[i] = cie_d65_10nm[i10] * (1.0 - f10) + cie_d65_10nm[i10 + 1u] * f10;
    w_trap[i] = (i == 0u || i + 1u == spectral_cie_sample_count) ? 0.5 : 1.0;
  }

  double y_int = 0.0;
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    y_int += w_trap[i] * cmf_1nm[i].y;
  }

  double y_d65_int = 0.0;
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    y_d65_int += w_trap[i] * cmf_1nm[i].y * d65_1nm[i];
  }
  const double d65_norm_scale = y_int / y_d65_int;

  Rgb xyz_d65{};
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    d65_1nm[i] *= d65_norm_scale;
    const double wt = w_trap[i] * d65_1nm[i] / y_int;
    xyz_d65.x += wt * cmf_1nm[i].x;
    xyz_d65.y += wt * cmf_1nm[i].y;
    xyz_d65.z += wt * cmf_1nm[i].z;
  }

  // Calibrate working-space XYZ->RGB matrix so (1,1,1) white illuminant maps
  // to exact (1,1,1) RGB.
  const auto m_rgb_to_xyz = invert_3x3(color_space);
  const std::array<double, 3u> xyz_white_target{
      m_rgb_to_xyz[0][0] + m_rgb_to_xyz[0][1] + m_rgb_to_xyz[0][2],
      m_rgb_to_xyz[1][0] + m_rgb_to_xyz[1][1] + m_rgb_to_xyz[1][2],
      m_rgb_to_xyz[2][0] + m_rgb_to_xyz[2][1] + m_rgb_to_xyz[2][2],
  };
  const double adapt_x = xyz_white_target[0] / std::max(xyz_d65.x, 1.0e-12);
  const double adapt_y = xyz_white_target[1] / std::max(xyz_d65.y, 1.0e-12);
  const double adapt_z = xyz_white_target[2] / std::max(xyz_d65.z, 1.0e-12);

  SpectralTablesHost out;
  out.spec_xyz_to_r = luisa::make_float3(
      static_cast<float>(color_space.xyz_to_r.x * adapt_x),
      static_cast<float>(color_space.xyz_to_r.y * adapt_y),
      static_cast<float>(color_space.xyz_to_r.z * adapt_z));
  out.spec_xyz_to_g = luisa::make_float3(
      static_cast<float>(color_space.xyz_to_g.x * adapt_x),
      static_cast<float>(color_space.xyz_to_g.y * adapt_y),
      static_cast<float>(color_space.xyz_to_g.z * adapt_z));
  out.spec_xyz_to_b = luisa::make_float3(
      static_cast<float>(color_space.xyz_to_b.x * adapt_x),
      static_cast<float>(color_space.xyz_to_b.y * adapt_y),
      static_cast<float>(color_space.xyz_to_b.z * adapt_z));
  out.rgb_to_y = luisa::make_float3(static_cast<float>(m_rgb_to_xyz[1][0]),
                                    static_cast<float>(m_rgb_to_xyz[1][1]),
                                    static_cast<float>(m_rgb_to_xyz[1][2]));

  out.cie_illuminant.resize(spectral_cie_sample_count);
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    out.cie_illuminant[i] = luisa::make_float4(
        static_cast<float>(cmf_1nm[i].x / y_int),
        static_cast<float>(cmf_1nm[i].y / y_int),
        static_cast<float>(cmf_1nm[i].z / y_int),
        static_cast<float>(d65_1nm[i]));
  }

  // 2. Build 3 x 96^3 Jakob & Hanika (2019) sigmoid-polynomial LUT.
  QuadratureWorkspace ws;
  ws.rgb_to_xyz = m_rgb_to_xyz;
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    ws.t[i] = static_cast<double>(i) /
              static_cast<double>(spectral_cie_sample_count - 1u);
    const double wt = w_trap[i] * d65_1nm[i] / y_int;
    ws.wx[i] = wt * cmf_1nm[i].x * adapt_x;
    ws.wy[i] = wt * cmf_1nm[i].y * adapt_y;
    ws.wz[i] = wt * cmf_1nm[i].z * adapt_z;
  }

  constexpr std::uint32_t res = rgb2spec_table_resolution;
  out.rgb2spec_coeffs.resize(static_cast<std::size_t>(3u) * res * res * res);
  std::array<double, res> z_grid{};
  for (std::uint32_t k = 0u; k < res; ++k) {
    double z = smoothstep(static_cast<double>(k) / static_cast<double>(res - 1u));
    z = std::clamp(z, 1.0e-5, 0.99995);
    z_grid[k] = z;
  }

  constexpr std::uint32_t mid = res / 2u;
  const std::size_t total_columns = static_cast<std::size_t>(3u) * res * res;
  psycles::parallel_for(total_columns, [&](std::size_t col) {
    const std::uint32_t face = static_cast<std::uint32_t>(col / (res * res));
    const std::uint32_t rem = static_cast<std::uint32_t>(col % (res * res));
    const std::uint32_t ix = rem / res;
    const std::uint32_t iy = rem % res;
    const double cx =
        smoothstep(static_cast<double>(ix) / static_cast<double>(res - 1u));
    const double cy =
        smoothstep(static_cast<double>(iy) / static_cast<double>(res - 1u));

    auto write_cell = [&](std::uint32_t k,
                          const std::array<double, 3u> &c) noexcept {
      const std::size_t idx =
          (((static_cast<std::size_t>(face) * res + ix) * res + iy) * res + k);
      out.rgb2spec_coeffs[idx] = luisa::make_float4(
          static_cast<float>(c[0]), static_cast<float>(c[1]),
          static_cast<float>(c[2]), 0.0f);
    };

    auto solve_k = [&](std::uint32_t k, std::array<double, 3u> &c) noexcept {
      const double z = z_grid[k];
      std::array<double, 3u> rgb{};
      rgb[face] = z;
      rgb[(face + 1u) % 3u] = z * cx;
      rgb[(face + 2u) % 3u] = z * cy;
      solve_sigmoid_poly(ws, rgb, c);
      write_cell(k, c);
    };

    std::array<double, 3u> c_mid{0.0, 0.0, 0.0};
    solve_k(mid, c_mid);

    std::array<double, 3u> c_up = c_mid;
    for (std::uint32_t k = mid + 1u; k < res; ++k) {
      solve_k(k, c_up);
    }

    std::array<double, 3u> c_down = c_mid;
    for (std::uint32_t k = mid; k > 0u; --k) {
      solve_k(k - 1u, c_down);
    }
  });

  // 3. Build 256-entry log-T Planck blackbody Y-normalization table.
  out.blackbody_norm.resize(blackbody_lut_count);
  const double log_min = std::log(static_cast<double>(blackbody_temp_min));
  const double log_max = std::log(static_cast<double>(blackbody_temp_max));
  for (std::uint32_t k = 0u; k < blackbody_lut_count; ++k) {
    const double u =
        static_cast<double>(k) / static_cast<double>(blackbody_lut_count - 1u);
    const double temp_k = std::exp(log_min + u * (log_max - log_min));
    double y_bb = 0.0;
    for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
      const double wl = static_cast<double>(spectral_lambda_min) +
                        static_cast<double>(i);
      const double spd = planck_relative_peak_normalized(wl, temp_k);
      y_bb += w_trap[i] * cmf_1nm[i].y * spd;
    }
    out.blackbody_norm[k] = static_cast<float>(y_int / std::max(y_bb, 1.0e-20));
  }

  try_save_cached(cache_path, cs_hash, out);
  fill_xyz3_tables(out);
  return out;
}

std::shared_ptr<SpectralTablesGpu> upload_spectral_tables(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const contract::ShaderColorSpace &color_space) {
  auto host = build_spectral_tables_host(color_space);
  auto gpu = std::make_shared<SpectralTablesGpu>();
  gpu->spec_xyz_to_r = host.spec_xyz_to_r;
  gpu->spec_xyz_to_g = host.spec_xyz_to_g;
  gpu->spec_xyz_to_b = host.spec_xyz_to_b;
  gpu->rgb_to_y = host.rgb_to_y;
  gpu->xyz3_integral_ratio = host.xyz3_integral_ratio;
  gpu->xyz3_to_r = host.xyz3_to_r;
  gpu->xyz3_to_g = host.xyz3_to_g;
  gpu->xyz3_to_b = host.xyz3_to_b;
  gpu->cie_illuminant_buffer =
      device.create_buffer<luisa::float4>(host.cie_illuminant.size());
  gpu->rgb2spec_buffer =
      device.create_buffer<luisa::float4>(host.rgb2spec_coeffs.size());
  gpu->blackbody_norm_buffer =
      device.create_buffer<float>(host.blackbody_norm.size());
  gpu->xyz3_cdfinv_buffer =
      device.create_buffer<luisa::float4>(host.xyz3_cdfinv.size());
  gpu->xyz3_collapse_buffer =
      device.create_buffer<luisa::float4>(host.xyz3_collapse.size());
  stream << gpu->cie_illuminant_buffer.copy_from(
                luisa::span<const luisa::float4>{host.cie_illuminant})
         << gpu->rgb2spec_buffer.copy_from(
                luisa::span<const luisa::float4>{host.rgb2spec_coeffs})
         << gpu->blackbody_norm_buffer.copy_from(
                luisa::span<const float>{host.blackbody_norm})
         << gpu->xyz3_cdfinv_buffer.copy_from(
                luisa::span<const luisa::float4>{host.xyz3_cdfinv})
         << gpu->xyz3_collapse_buffer.copy_from(
                luisa::span<const luisa::float4>{host.xyz3_collapse})
         << luisa::compute::synchronize();
  return gpu;
}

float sample_visible_wavelength_host(float u) noexcept {
  const double uc = std::clamp(static_cast<double>(u), 0.0, 1.0);
  const double t =
      std::clamp(wavelength_sample_t_lo + wavelength_sample_span * uc,
                 -0.99999999, 0.99999999);
  const double wl = wavelength_sample_center +
                    std::atanh(t) / wavelength_sample_a;
  return static_cast<float>(
      std::clamp(wl, static_cast<double>(spectral_lambda_min),
                 static_cast<double>(spectral_lambda_max)));
}

float visible_wavelength_pdf_host(float lambda_nm) noexcept {
  if (lambda_nm < spectral_lambda_min || lambda_nm > spectral_lambda_max) {
    return 0.0f;
  }
  const double c = std::cosh(wavelength_sample_a *
                             (static_cast<double>(lambda_nm) -
                              wavelength_sample_center));
  return static_cast<float>(wavelength_sample_pdf_scale / (c * c));
}

std::array<float, 3u> eval_rgb2spec_coeffs_host(
    const SpectralTablesHost &tables, luisa::float3 rgb) noexcept {
  const float r = std::max(rgb.x, 0.0f);
  const float g = std::max(rgb.y, 0.0f);
  const float b = std::max(rgb.z, 0.0f);
  std::uint32_t face = 0u;
  float m = r;
  float c1 = 0.0f;
  float c2 = 0.0f;
  if (r >= g && r >= b) {
    face = 0u;
    m = r;
    c1 = m > 0.0f ? g / m : 0.0f;
    c2 = m > 0.0f ? b / m : 0.0f;
  } else if (g >= r && g >= b) {
    face = 1u;
    m = g;
    c1 = m > 0.0f ? b / m : 0.0f;
    c2 = m > 0.0f ? r / m : 0.0f;
  } else {
    face = 2u;
    m = b;
    c1 = m > 0.0f ? r / m : 0.0f;
    c2 = m > 0.0f ? g / m : 0.0f;
  }

  constexpr double scale_n = static_cast<double>(rgb2spec_table_resolution - 1u);
  const double u = inv_smoothstep(static_cast<double>(c1)) * scale_n;
  const double v = inv_smoothstep(static_cast<double>(c2)) * scale_n;
  const double w = inv_smoothstep(static_cast<double>(m)) * scale_n;

  const auto clamp_idx = [](double val) noexcept {
    const std::uint32_t max_i = rgb2spec_table_resolution - 2u;
    const auto i = static_cast<std::uint32_t>(
        std::clamp(std::floor(val), 0.0, static_cast<double>(max_i)));
    const double f = std::clamp(val - static_cast<double>(i), 0.0, 1.0);
    return std::pair{i, f};
  };
  const auto [ix, fx] = clamp_idx(u);
  const auto [iy, fy] = clamp_idx(v);
  const auto [iz, fz] = clamp_idx(w);

  constexpr std::size_t res = rgb2spec_table_resolution;
  auto at = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return tables.rgb2spec_coeffs[(((static_cast<std::size_t>(face) * res + x) *
                                        res +
                                    y) *
                                   res +
                                   z)];
  };
  std::array<double, 3u> out{0.0, 0.0, 0.0};
  for (std::uint32_t dx = 0u; dx < 2u; ++dx) {
    const double wx = dx ? fx : (1.0 - fx);
    for (std::uint32_t dy = 0u; dy < 2u; ++dy) {
      const double wy = dy ? fy : (1.0 - fy);
      for (std::uint32_t dz = 0u; dz < 2u; ++dz) {
        const double wz = dz ? fz : (1.0 - fz);
        const double weight = wx * wy * wz;
        const auto cell = at(ix + dx, iy + dy, iz + dz);
        out[0] += weight * cell.x;
        out[1] += weight * cell.y;
        out[2] += weight * cell.z;
      }
    }
  }
  return {static_cast<float>(out[0]), static_cast<float>(out[1]),
          static_cast<float>(out[2])};
}

luisa::float4 eval_rgb2spec_reflectance_host(
    const SpectralTablesHost &tables, luisa::float3 rgb,
    luisa::float4 lambdas) noexcept {
  float r = std::max(rgb.x, 0.0f);
  float g = std::max(rgb.y, 0.0f);
  float b = std::max(rgb.z, 0.0f);
  float m = std::max(r, std::max(g, b));
  const float mn = std::min(r, std::min(g, b));
  if (m - mn <= 1.0e-6f * std::max(m, 1.0f)) {
    return luisa::make_float4(m);
  }
  float base = 0.0f;
  float scale = 1.0f;
  if (mn > 0.5f && m <= 1.0f && m > 0.9f) {
    base = mn;
    scale = 1.0f - mn;
    r = (r - base) / scale;
    g = (g - base) / scale;
    b = (b - base) / scale;
  } else if (m > 1.0f) {
    scale = 2.0f * m;
    r /= scale;
    g /= scale;
    b /= scale;
  }
  const auto c = eval_rgb2spec_coeffs_host(tables, luisa::make_float3(r, g, b));
  auto eval_lane = [&](float wl) noexcept -> float {
    const float t =
        (std::clamp(wl, spectral_lambda_min, spectral_lambda_max) -
         spectral_lambda_min) /
        spectral_lambda_span;
    const float p = (c[0] * t + c[1]) * t + c[2];
    return base + scale * (0.5f + 0.5f * p / std::sqrt(1.0f + p * p));
  };
  return luisa::make_float4(eval_lane(lambdas.x), eval_lane(lambdas.y),
                            eval_lane(lambdas.z), eval_lane(lambdas.w));
}

luisa::float3 integrate_reflectance_xyz_host(
    const SpectralTablesHost &tables, luisa::float3 rgb) noexcept {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  for (std::size_t i = 0u; i < spectral_cie_sample_count; ++i) {
    const float wl = spectral_lambda_min + static_cast<float>(i);
    const float refl = eval_rgb2spec_reflectance_host(
                           tables, rgb, luisa::make_float4(wl)).x;
    const double wt =
        (i == 0u || i + 1u == spectral_cie_sample_count) ? 0.5 : 1.0;
    const auto cie = tables.cie_illuminant[i];
    const double contrib = wt * static_cast<double>(refl) *
                           static_cast<double>(cie.w);
    x += contrib * static_cast<double>(cie.x);
    y += contrib * static_cast<double>(cie.y);
    z += contrib * static_cast<double>(cie.z);
  }
  return luisa::make_float3(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(z));
}

luisa::float3 sample_xyz3_lambdas_host(
    const SpectralTablesHost &tables, float u) noexcept {
  const double pos =
      std::clamp(static_cast<double>(u), 0.0, 1.0) *
      static_cast<double>(xyz3_cdfinv_lut_count - 1u);
  const std::size_t i0 = std::min(static_cast<std::size_t>(pos),
                                  static_cast<std::size_t>(xyz3_cdfinv_lut_count - 2u));
  const float frac = static_cast<float>(pos - static_cast<double>(i0));
  const auto a = tables.xyz3_cdfinv[i0];
  const auto b = tables.xyz3_cdfinv[i0 + 1u];
  return luisa::make_float3(
      std::clamp(a.x + (b.x - a.x) * frac, spectral_lambda_min, spectral_lambda_max),
      std::clamp(a.y + (b.y - a.y) * frac, spectral_lambda_min, spectral_lambda_max),
      std::clamp(a.z + (b.z - a.z) * frac, spectral_lambda_min, spectral_lambda_max));
}

luisa::float3 xyz3_collapse_weight_host(
    const SpectralTablesHost &tables, float lambda_nm) noexcept {
  const double t =
      std::clamp((static_cast<double>(lambda_nm) - spectral_lambda_min) /
                     static_cast<double>(spectral_lambda_span),
                 0.0, 1.0) *
      static_cast<double>(xyz3_collapse_lut_count - 1u);
  const std::size_t i0 = std::min(static_cast<std::size_t>(t),
                                  static_cast<std::size_t>(xyz3_collapse_lut_count - 2u));
  const float frac = static_cast<float>(t - static_cast<double>(i0));
  const auto a = tables.xyz3_collapse[i0];
  const auto b = tables.xyz3_collapse[i0 + 1u];
  return luisa::make_float3(a.x + (b.x - a.x) * frac,
                            a.y + (b.y - a.y) * frac,
                            a.z + (b.z - a.z) * frac);
}

luisa::float3 xyz3_pdf_host(
    const SpectralTablesHost &tables, float lambda_nm) noexcept {
  if (lambda_nm < spectral_lambda_min || lambda_nm > spectral_lambda_max) {
    return luisa::make_float3(0.0f);
  }
  const double pos = std::clamp(static_cast<double>(lambda_nm - spectral_lambda_min),
                                0.0, static_cast<double>(spectral_lambda_span));
  const std::size_t i0 = std::min(static_cast<std::size_t>(pos),
                                  static_cast<std::size_t>(spectral_cie_sample_count - 2u));
  const double frac = pos - static_cast<double>(i0);
  const auto c0 = tables.cie_illuminant[i0];
  const auto c1 = tables.cie_illuminant[i0 + 1u];
  const double cx = std::max(0.0, c0.x * (1.0 - frac) + c1.x * frac);
  const double cy = std::max(0.0, c0.y * (1.0 - frac) + c1.y * frac);
  const double cz = std::max(0.0, c0.z * (1.0 - frac) + c1.z * frac);
  const auto &ratio = tables.xyz3_integral_ratio;
  return luisa::make_float3(static_cast<float>(cx / std::max<double>(ratio.x, 1.0e-30)),
                            static_cast<float>(cy / std::max<double>(ratio.y, 1.0e-30)),
                            static_cast<float>(cz / std::max<double>(ratio.z, 1.0e-30)));
}

luisa::float3 xyz3_to_rgb_host(
    const SpectralTablesHost &tables, luisa::float3 raw) noexcept {
  auto d = [](luisa::float3 a, luisa::float3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  };
  return luisa::make_float3(d(tables.xyz3_to_r, raw),
                            d(tables.xyz3_to_g, raw),
                            d(tables.xyz3_to_b, raw));
}

} // namespace psycles::luisa_backend::cycles_spectrum
