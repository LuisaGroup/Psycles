#pragma once

#include <cstddef>

namespace psycles::luisa_backend::cycles52_thin_film_tables {

inline constexpr std::size_t thin_film_cmf_row_count = 512u;
inline constexpr std::size_t thin_film_cmf_component_count = 6u;

// Cycles 5.2.1 source CMFs remain immutable typed host data. The definition
// lives in one translation unit so consumers do not textually instantiate the
// generated table or evade the ordinary C++ dependency boundary.
extern const float
    table_thin_film_cmf[thin_film_cmf_row_count]
                       [thin_film_cmf_component_count];

} // namespace psycles::luisa_backend::cycles52_thin_film_tables
