#ifndef TOOLS__OPENVINO_UTILS_HPP
#define TOOLS__OPENVINO_UTILS_HPP

#include <openvino/openvino.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/logger.hpp"

namespace tools::ov_utils
{
inline std::string join(const std::vector<std::string> & items, const char * sep = ", ")
{
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      oss << sep;
    }
    oss << items[i];
  }
  return oss.str();
}

inline std::vector<std::string> build_compile_candidates(const std::string & requested_device)
{
  const std::string preferred_device = requested_device.empty() ? "AUTO:GPU,CPU" : requested_device;
  std::vector<std::string> candidates;

  const auto append_unique = [&candidates](const std::string & device) {
      if (device.empty()) {
        return;
      }
      if (std::find(candidates.begin(), candidates.end(), device) == candidates.end()) {
        candidates.push_back(device);
      }
    };

  append_unique(preferred_device);

  if (preferred_device != "AUTO:GPU,CPU" && preferred_device.find("GPU") != std::string::npos) {
    append_unique("AUTO:GPU,CPU");
  }

  if (preferred_device != "CPU") {
    append_unique("CPU");
  }

  return candidates;
}

inline void log_available_devices(ov::Core & core, const std::string & module_name)
{
  try {
    const auto available_devices = core.get_available_devices();
    if (available_devices.empty()) {
      tools::logger()->warn("[{}] OpenVINO did not report any available devices", module_name);
      return;
    }

    tools::logger()->info(
      "[{}] OpenVINO available devices: {}", module_name, join(available_devices));

    for (const auto & device : available_devices) {
      try {
        const auto full_name = core.get_property(device, ov::device::full_name);
        tools::logger()->info("[{}] {} => {}", module_name, device, full_name);
      } catch (const std::exception & e) {
        tools::logger()->warn(
          "[{}] Failed to query FULL_DEVICE_NAME for {}: {}", module_name, device, e.what());
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->warn(
      "[{}] Failed to query OpenVINO available devices: {}", module_name, e.what());
  }
}

inline void log_execution_devices(
  const ov::CompiledModel & compiled_model, const std::string & module_name,
  const std::string & requested_device, const std::string & compiled_device)
{
  try {
    const auto execution_devices = compiled_model.get_property(ov::execution_devices);
    tools::logger()->info(
      "[{}] OpenVINO requested={}, compiled={}, execution devices={}", module_name,
      requested_device, compiled_device, join(execution_devices));
  } catch (const std::exception & e) {
    tools::logger()->warn(
      "[{}] Failed to query execution devices after compiling {}: {}", module_name,
      compiled_device, e.what());
  }
}

template<typename... Properties>
inline ov::CompiledModel compile_model_with_fallback(
  ov::Core & core, const std::shared_ptr<ov::Model> & model, const std::string & requested_device,
  const std::string & module_name, const Properties &... properties)
{
  log_available_devices(core, module_name);
  const auto compile_candidates = build_compile_candidates(requested_device);

  std::string last_error = "unknown error";
  for (const auto & device : compile_candidates) {
    try {
      auto compiled_model = core.compile_model(model, device, properties...);
      log_execution_devices(compiled_model, module_name, requested_device, device);
      return compiled_model;
    } catch (const std::exception & e) {
      last_error = e.what();
      tools::logger()->warn(
        "[{}] Failed to compile model on {}: {}", module_name, device, e.what());
    }
  }

  throw std::runtime_error(
          "[" + module_name + "] Failed to compile model. Last error: " + last_error);
}
}  // namespace tools::ov_utils

#endif  // TOOLS__OPENVINO_UTILS_HPP
