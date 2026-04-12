#ifndef TOOLS__RUNTIME_PARAMS_HPP
#define TOOLS__RUNTIME_PARAMS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace tools::runtime_params
{

void register_config(const std::string & config_path);
bool is_registered(const std::string & config_path);
uint64_t version(const std::string & config_path);

double get_double(const std::string & config_path, const std::string & key);
int get_int(const std::string & config_path, const std::string & key);
bool get_bool(const std::string & config_path, const std::string & key);
std::string get_string(const std::string & config_path, const std::string & key);
std::vector<double> get_number_array(const std::string & config_path, const std::string & key);

nlohmann::json describe(const std::string & config_path);
nlohmann::json apply(
  const std::string & config_path, const nlohmann::json & request,
  const std::string & source = "web");
nlohmann::json reset(
  const std::string & config_path, const std::vector<std::string> & keys = {},
  const std::string & source = "web");

}  // namespace tools::runtime_params

#endif  // TOOLS__RUNTIME_PARAMS_HPP
