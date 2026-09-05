#pragma once

// Post-decision rune truth association.  Detector/classifier/Target/Aimer never call this.
// In `all` mode the routing key is the algorithm's selected mode; truth is scored afterwards
// by (wanted_mode + nearest r_center_odom among runes of that mode).  A dual small+big scene
// is therefore not inherently ambiguous.  Hits>1 of the SAME mode still is.

#include <cstddef>
#include <cstdint>

#include <Eigen/Dense>

#include "simulation/io/shared_memory_layout.hpp"

namespace sim_io
{

struct RuneAssociation
{
  const GroundTruthRune * selected = nullptr;
  std::size_t hits = 0;
  bool ambiguous = false;
};

// wanted_mode is the GroundTruthRune::rune_mode wire value: 0=small, 1=big.
// Negative wanted_mode means the algorithm has not routed a decision; no association.
// estimate_center, when non-null, selects the nearest center among same-mode hits.
inline RuneAssociation associate_rune_by_mode(
  const GroundTruthRune * runes, std::uint32_t rune_count, int wanted_mode,
  const Eigen::Vector3d * estimate_center)
{
  RuneAssociation out;
  if (runes == nullptr || wanted_mode < 0) return out;
  const std::uint32_t n =
    rune_count < static_cast<std::uint32_t>(GROUND_TRUTH_MAX_RUNES)
      ? rune_count
      : static_cast<std::uint32_t>(GROUND_TRUTH_MAX_RUNES);
  double best_d2 = 0.0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto & rune = runes[i];
    if (static_cast<int>(rune.rune_mode) != wanted_mode) continue;
    ++out.hits;
    const Eigen::Vector3d center(
      static_cast<double>(rune.r_center_odom[0]),
      static_cast<double>(rune.r_center_odom[1]),
      static_cast<double>(rune.r_center_odom[2]));
    if (out.selected == nullptr) {
      out.selected = &rune;
      if (estimate_center) best_d2 = (*estimate_center - center).squaredNorm();
      continue;
    }
    if (estimate_center) {
      const double d2 = (*estimate_center - center).squaredNorm();
      if (d2 < best_d2) {
        out.selected = &rune;
        best_d2 = d2;
      }
    }
  }
  out.ambiguous = out.hits > 1;
  return out;
}

}  // namespace sim_io
