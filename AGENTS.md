# sp_vision25 Contributor Guide

## Scope and Priorities

This repository is a RoboMaster vision system. It contains time-sensitive auto aim,
ballistic planning, OpenVINO inference, serial I/O, ROS2 integration, and an in-process
web debugger. Prefer correctness of coordinates, timestamps, state-estimation invariants,
and bounded latency over broad refactors or cosmetic changes.

Before changing code, identify the concrete runtime entry point and trace only its direct
callers, callees, config keys, and focused tests. Use `rg` for source discovery. Do not read
generated build outputs, model binaries, recordings, or SDK/vendor folders unless the task is
specifically about them.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `tasks/auto_aim/` | Armor detection, PnP/reprojection, tracker, 13D target model, planner, shooter |
| `tasks/auto_aim/multithread/` | Bounded OpenVINO asynchronous detection path used by `mt_standard` |
| `tasks/auto_buff/` | Power-rune detection, tracking, and aiming |
| `tools/` | EKF, math, runtime parameters, queues, debug JSON, web server, paths |
| `io/` | Camera, serial, gimbal, IMU, and hardware adapters |
| `src/` | Runtime entry points such as `standard_mpc`, `mt_standard`, and debug programs |
| `configs/` | Runtime YAML for demo, standard robot, and sentry deployments |
| `assets/web_debugger/` | Static web debugger HTML/CSS/JavaScript |
| `tests/` | Offline, focused, camera, planner, and system tests |
| `docs/` | Engineering docs, parameter mapping, and Awakening comparison |

## Build and Test

Build outside source directories. The direct CMake route works without ROS2-specific targets:

```bash
cmake -S . -B build
cmake --build build --target auto_aim standard_mpc mt_standard auto_aim_test_web -j2
ctest --test-dir build --output-on-failure
```

For changes to EKF, queues, target state, projection, or observation logic, run both focused
tests after building them:

```bash
cmake --build build --target estimator_pipeline_test target_13d_test -j2
ctest --test-dir build --output-on-failure
```

ROS2 targets are enabled only when all required packages, including `sp_msgs`, are discoverable.
Do not report sentry/ROS2 validation as complete when CMake prints that ROS2-specific code was
skipped. Use the workspace's normal `colcon build` workflow only when the ROS2 environment has
been sourced and the dependent messages are available.

For a quick offline Web check, run from the repository root so config and assets resolve
correctly:

```bash
build/auto_aim_test_web assets/demo/demo \
  --config-path=configs/demo.yaml --show-local=false \
  --web-host=127.0.0.1 --web-port=8090
```

The web debugger exposes `GET /healthz`, `GET /api/state`, `GET /api/params`, `GET /data`, and
MJPEG/JPEG frame endpoints. Keep a server bound to loopback during local development unless a
trusted robot LAN explicitly requires another address.

## Auto-Aim State Estimation Invariants

`auto_aim::Target` uses the 13D storage order below. Do not use raw indexes in new code; use
`target_state::Index` names.

```text
[cx, vx, cy, vy, cz, vz, rot_z, vyaw, log_r1, log_r2, h, rot_y, rot_x]
```

- `rot_x`, `rot_y`, and `rot_z` form one SO(3) rotation vector. They are not independent Euler
  angles. Use `tools::so3_exp()` and `tools::so3_log()`.
- The error-state convention is right perturbation:
  `R_new = R * Exp(delta_rot)` and
  `delta_rot = Log(R_nominal.transpose() * R_value)`.
- Use `Target::car_rpy()` only for display. Use `Target::radius()` instead of treating
  `LOG_R1` or `LOG_R2` as a linear radius.
- Normal targets estimate full RPY. Outpost and base intentionally use yaw-only rotation.
  Preserve the outpost three-board height-offset mapping and balance-infantry two-board model.
- Prediction has yaw angular velocity only. Roll and pitch are random walks, so the current 13D
  model follows, but does not predict, fast roll/pitch motion. Adding roll/pitch angular velocity
  is an explicit model expansion requiring new state slots, Q, Jacobians, tests, and A/B data.
- The filter observes a complete armor as two UVL lightbars, eight values total. Angle residuals
  must be wrapped. The default 99% NIS gate is `20.090` for 8 DoF.
- A rejected measurement must not mutate state, covariance, `last_id`, jump/switch bookkeeping,
  or convergence count. Maintain finite-value checks before EKF updates and retain Joseph-form
  covariance updates plus symmetry enforcement.

## Coordinates, Projection, and Time

- Keep coordinate-frame ownership explicit: armor pose, gimbal pose, camera pose, and world pose
  must not be mixed. `Armor::R_armor2world` is the full PnP orientation used for initialization.
- Use the matrix overload of `Solver::reproject_armor()` when projecting a full rigid-body pose.
  The yaw-only overload is retained for compatibility and must not discard a full RPY estimate.
- Preserve `std::chrono::steady_clock` timestamps through camera, inference, tracker, and planner
  paths. Treat frame age as an observable performance metric, not just inference FPS.
- YAML angles and user-facing MPC accelerations are generally in degrees; internal planner and
  filter math is radians. `max_yaw_acc` and `max_pitch_acc` are converted from `deg/s^2` to
  `rad/s^2` before TinyMPC. Do not remove that conversion.

## Concurrency Rules

- Queue capacity is part of the latency contract. Use `try_push`, `pop_for`, and `clear` rather
  than composing `empty`, `front`, and `pop` across threads.
- Inference overload should drop work before allocating/starting a request, update telemetry, and
  preserve the latest useful frame. Do not introduce unbounded queues.
- When changing mode or releasing inference resources, clear obsolete pending work and ensure
  request-owned input memory remains valid until inference completes.
- `inference_max_inflight` is currently a construction-time parameter for `MultiThreadDetector`.
  Mark or document restart/rebuild requirements rather than claiming a live Web edit reconfigures
  an existing request queue.

## Config, Web, and Documentation

- Add a new runtime parameter in all three places: `tools/runtime_params.cpp`, applicable YAML
  files, and the owning module's reload path. `WebDebugger` must be able to register every
  supported YAML; a missing required key disables the parameter panel.
- Keep `demo.yaml`, `standard3.yaml`, and `sentry.yaml` semantically aligned for shared tracker,
  estimator, inference, and debugger keys. Use explicit defaults where a parameter lacks a
  runtime-parameter fallback.
- State JSON is a frontend contract. When renaming a value, update the producer in `tools/debug.*`,
  the entry-point payload, `assets/web_debugger/static/js/main.js`, and chart metadata together.
- Web parameter writes have no authentication. Do not expose the debugger to an untrusted network.
- Update `docs/awakening_算法对比与优化指南.md` when a reference feature moves between
  "implemented" and "remaining". Never claim improved hit rate without controlled real-robot or
  truth-labeled A/B evidence.

## Editing and Git Hygiene

- Use C++17 and the existing project style. Keep edits scoped; do not reformat unrelated files.
- Preserve user changes in a dirty worktree. Do not reset, checkout, or discard unrelated files.
- Add focused tests in `tests/` for estimator, projection, queue, or planner behavior changes.
- Before a requested commit, run `git diff --check`, inspect `git diff --stat`, and ensure no
  generated `build/`, `install/`, `log/`, recordings, model binaries, or secrets are staged.
- Commit messages use concise Chinese imperative tags, for example:
  `[优化] 迁移13维刚体状态与UVL观测`.
