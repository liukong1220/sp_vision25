#ifndef SIMULATION_IO__REPORT_METADATA_HPP
#define SIMULATION_IO__REPORT_METADATA_HPP

#include <nlohmann/json.hpp>

#include <string>

namespace sim_io
{
std::string file_sha256(const std::string & path);
std::string utc_timestamp();
nlohmann::json process_command_line();
nlohmann::json git_repository_metadata(const std::string & path);
nlohmann::json reproducibility_metadata(
  const std::string & config_path, const std::string & model_path);
}  // namespace sim_io

#endif  // SIMULATION_IO__REPORT_METADATA_HPP
