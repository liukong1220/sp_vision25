#ifndef SIMULATION_IO__STRICT_CLOSED_LOOP_HPP
#define SIMULATION_IO__STRICT_CLOSED_LOOP_HPP

#include <cstdint>
#include <string>
#include <utility>

namespace sim_io
{

// Named conjuncts for the sim_auto_aim strict JSON verdict.
// PASS is the AND of every listed conjunct, including (shot_evidence OR no_shot_aiming).
// The criterion string is generated from this same list — never a stale hardcoded sentence.
struct StrictClosedLoopInputs
{
  bool enough_frames = false;
  bool enough_gt_fetches = false;
  bool enough_gt_coverage = false;
  bool no_gt_mismatch = false;
  bool no_gt_timestamp_mismatch = false;
  bool no_gt_ambiguous = false;
  bool no_gt_nearest = false;
  bool no_degraded_armor = false;
  bool enough_matched_samples = false;
  bool dynamic_budget_ok = false;
  bool controlled_motion_observed = false;
  bool closed_loop_mode = false;
  bool algorithm_chains = false;
  bool no_offending = false;
  bool no_suppressed_fires = false;
  bool color_gate = false;
  bool shot_evidence = false;
  bool no_shot_aiming = false;
};

struct StrictClosedLoopResult
{
  bool truth_contract = false;
  bool algorithm_closed_loop = false;
  bool shot_or_aiming = false;
  bool passed = false;
  std::string criterion;
};

// Pre-fix truth_contract: omitted ambiguous/nearest/degraded/enough_frames/enough_gt_fetches.
inline bool old_eval_truth_contract(
  bool do_eval,
  std::uint32_t attempts,
  std::uint32_t frames,
  std::uint32_t success,
  std::uint32_t mismatch,
  std::uint32_t timestamp_mismatch,
  bool coverage,
  bool matched_samples,
  std::uint32_t count)
{
  return do_eval && attempts == frames && success == frames && mismatch == 0 &&
    timestamp_mismatch == 0 && coverage && matched_samples && count > 0;
}

struct OldEvalTruthStats
{
  bool do_eval = true;
  std::uint32_t attempts = 0;
  std::uint32_t frames = 0;
  std::uint32_t success = 0;
  std::uint32_t mismatch = 0;
  std::uint32_t timestamp_mismatch = 0;
  bool coverage = false;
  bool matched_samples = false;
  std::uint32_t count = 0;
  std::uint32_t ambiguous = 0;
  std::uint32_t nearest = 0;
  std::uint32_t degraded = 0;
  std::uint32_t min_frames = 30;
};

inline bool old_eval_truth_contract(const OldEvalTruthStats & stats)
{
  return old_eval_truth_contract(
    stats.do_eval, stats.attempts, stats.frames, stats.success, stats.mismatch,
    stats.timestamp_mismatch, stats.coverage, stats.matched_samples, stats.count);
}

inline StrictClosedLoopInputs passing_strict_inputs()
{
  StrictClosedLoopInputs in;
  in.enough_frames = true;
  in.enough_gt_fetches = true;
  in.enough_gt_coverage = true;
  in.no_gt_mismatch = true;
  in.no_gt_timestamp_mismatch = true;
  in.no_gt_ambiguous = true;
  in.no_gt_nearest = true;
  in.no_degraded_armor = true;
  in.enough_matched_samples = true;
  in.dynamic_budget_ok = true;
  in.controlled_motion_observed = true;
  in.closed_loop_mode = true;
  in.algorithm_chains = true;
  in.no_offending = true;
  in.no_suppressed_fires = true;
  in.color_gate = true;
  in.shot_evidence = false;
  in.no_shot_aiming = true;
  return in;
}

inline StrictClosedLoopInputs inputs_from_old_stats(const OldEvalTruthStats & stats)
{
  StrictClosedLoopInputs in = passing_strict_inputs();
  in.enough_frames = stats.frames >= stats.min_frames;
  in.enough_gt_fetches = stats.attempts >= stats.min_frames;
  in.enough_gt_coverage = stats.coverage;
  in.no_gt_mismatch = stats.mismatch == 0;
  in.no_gt_timestamp_mismatch = stats.timestamp_mismatch == 0;
  in.no_gt_ambiguous = stats.ambiguous == 0;
  in.no_gt_nearest = stats.nearest == 0;
  in.no_degraded_armor = stats.degraded == 0;
  in.enough_matched_samples = stats.matched_samples;
  return in;
}

inline StrictClosedLoopResult evaluate_strict_closed_loop(const StrictClosedLoopInputs & in)
{
  const std::pair<const char *, bool> conjuncts[] = {
    {"enough_frames", in.enough_frames},
    {"enough_gt_fetches", in.enough_gt_fetches},
    {"enough_gt_coverage", in.enough_gt_coverage},
    {"no_gt_mismatch", in.no_gt_mismatch},
    {"no_gt_timestamp_mismatch", in.no_gt_timestamp_mismatch},
    {"no_gt_ambiguous", in.no_gt_ambiguous},
    {"no_gt_nearest", in.no_gt_nearest},
    {"no_degraded_armor", in.no_degraded_armor},
    {"enough_matched_samples", in.enough_matched_samples},
    {"dynamic_budget_ok", in.dynamic_budget_ok},
    {"controlled_motion_observed", in.controlled_motion_observed},
    {"closed_loop_mode", in.closed_loop_mode},
    {"algorithm_chains", in.algorithm_chains},
    {"offending==0", in.no_offending},
    {"suppressed_fires==0", in.no_suppressed_fires},
    {"color_gate", in.color_gate},
  };

  bool passed = true;
  std::string criterion;
  for (const auto & conjunct : conjuncts) {
    passed = passed && conjunct.second;
    if (!criterion.empty()) criterion += " AND ";
    criterion += conjunct.first;
  }
  const bool shot_or_aiming = in.shot_evidence || in.no_shot_aiming;
  passed = passed && shot_or_aiming;
  criterion += " AND (shot_evidence OR no_shot_aiming)";

  StrictClosedLoopResult out;
  out.shot_or_aiming = shot_or_aiming;
  out.passed = passed;
  out.criterion = criterion;
  out.truth_contract = in.enough_frames && in.enough_gt_fetches && in.enough_gt_coverage &&
    in.no_gt_mismatch && in.no_gt_timestamp_mismatch && in.no_gt_ambiguous &&
    in.no_gt_nearest && in.no_degraded_armor && in.enough_matched_samples;
  out.algorithm_closed_loop = in.closed_loop_mode && in.algorithm_chains &&
    in.dynamic_budget_ok && in.controlled_motion_observed && in.no_offending &&
    in.no_suppressed_fires && shot_or_aiming;
  return out;
}

}  // namespace sim_io

#endif  // SIMULATION_IO__STRICT_CLOSED_LOOP_HPP
