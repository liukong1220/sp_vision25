#include "report_metadata.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

namespace sim_io
{
namespace
{
struct ProcessResult
{
  int exit_code = -1;
  std::string output;
};

ProcessResult run_capture(
  const std::vector<std::string> & arguments,
  const std::optional<std::string_view> & input = std::nullopt)
{
  if (arguments.empty()) return {};
  int stdout_pipe[2];
  int stdin_pipe[2] = {-1, -1};
  if (::pipe(stdout_pipe) != 0 || (input.has_value() && ::pipe(stdin_pipe) != 0)) return {};

  const pid_t pid = ::fork();
  if (pid == 0) {
    ::dup2(stdout_pipe[1], STDOUT_FILENO);
    if (input.has_value()) ::dup2(stdin_pipe[0], STDIN_FILENO);
    const int dev_null = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (dev_null >= 0) ::dup2(dev_null, STDERR_FILENO);
    ::close(stdout_pipe[0]);
    ::close(stdout_pipe[1]);
    if (stdin_pipe[0] >= 0) ::close(stdin_pipe[0]);
    if (stdin_pipe[1] >= 0) ::close(stdin_pipe[1]);

    std::vector<char *> argv;
    argv.reserve(arguments.size() + 1);
    for (const auto & argument : arguments)
      argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    ::execvp(argv[0], argv.data());
    std::_Exit(127);
  }

  ::close(stdout_pipe[1]);
  if (stdin_pipe[0] >= 0) ::close(stdin_pipe[0]);
  if (pid < 0) {
    ::close(stdout_pipe[0]);
    if (stdin_pipe[1] >= 0) ::close(stdin_pipe[1]);
    return {};
  }

  if (input.has_value()) {
    std::size_t written = 0;
    while (written < input->size()) {
      const ssize_t n = ::write(stdin_pipe[1], input->data() + written, input->size() - written);
      if (n <= 0) break;
      written += static_cast<std::size_t>(n);
    }
    ::close(stdin_pipe[1]);
  }

  ProcessResult result;
  char buffer[4096];
  for (;;) {
    const ssize_t n = ::read(stdout_pipe[0], buffer, sizeof(buffer));
    if (n <= 0) break;
    result.output.append(buffer, static_cast<std::size_t>(n));
  }
  ::close(stdout_pipe[0]);
  int status = 0;
  if (::waitpid(pid, &status, 0) == pid && WIFEXITED(status))
    result.exit_code = WEXITSTATUS(status);
  return result;
}

std::string trim_newline(std::string value)
{
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
  return value;
}

std::string hash_bytes(std::string_view bytes)
{
  char name[] = "/tmp/sim_report_hash_XXXXXX";
  const int fd = ::mkstemp(name);
  if (fd < 0) return {};
  std::size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
    if (n <= 0) {
      ::close(fd);
      ::unlink(name);
      return {};
    }
    written += static_cast<std::size_t>(n);
  }
  ::close(fd);
  const auto result = run_capture({"sha256sum", "--", name});
  ::unlink(name);
  if (result.exit_code != 0) return {};
  const auto delimiter = result.output.find_first_of(" \t\r\n");
  return result.output.substr(0, delimiter);
}

std::string read_binary_file(const std::filesystem::path & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::ostringstream data;
  data << input.rdbuf();
  return data.str();
}

std::vector<std::string> split_nul(const std::string & value)
{
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const auto end = value.find('\0', begin);
    fields.emplace_back(value.substr(begin, end == std::string::npos ? end : end - begin));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return fields;
}
}  // namespace

std::string file_sha256(const std::string & path)
{
  if (path.empty() || !std::filesystem::is_regular_file(path)) return {};
  const auto result = run_capture({"sha256sum", "--", path});
  if (result.exit_code != 0) return {};
  const auto delimiter = result.output.find_first_of(" \t\r\n");
  return result.output.substr(0, delimiter);
}

std::string utc_timestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  ::gmtime_r(&now, &utc);
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return text;
}

nlohmann::json process_command_line()
{
  const std::string bytes = read_binary_file("/proc/self/cmdline");
  nlohmann::json arguments = nlohmann::json::array();
  for (const auto & argument : split_nul(bytes))
    if (!argument.empty()) arguments.push_back(argument);
  return arguments;
}

nlohmann::json git_repository_metadata(const std::string & path)
{
  const auto git = [&](std::initializer_list<const char *> tail) {
    std::vector<std::string> args{"git", "-C", path};
    for (const char * value : tail) args.emplace_back(value);
    return run_capture(args);
  };

  const auto head = git({"rev-parse", "HEAD"});
  const auto branch = git({"branch", "--show-current"});
  const auto upstream = git({"rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"});
  const auto divergence = git({"rev-list", "--left-right", "--count", "HEAD...@{upstream}"});
  const auto status = git({"status", "--porcelain=v1", "-z", "--untracked-files=all"});
  const auto tracked_diff = git({"diff", "--binary", "HEAD", "--"});

  nlohmann::json dirty_files = nlohmann::json::array();
  nlohmann::json untracked_files = nlohmann::json::array();
  std::vector<std::string> untracked_paths;
  const auto records = split_nul(status.output);
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto & record = records[i];
    if (record.size() < 4) continue;
    const std::string code = record.substr(0, 2);
    const std::string file = record.substr(3);
    dirty_files.push_back({{"status", code}, {"path", file}});
    if (code == "??") {
      untracked_files.push_back(file);
      untracked_paths.push_back(file);
    }
    if ((code[0] == 'R' || code[0] == 'C') && i + 1 < records.size()) {
      dirty_files.push_back({{"status", "source"}, {"path", records[++i]}});
    }
  }

  std::string canonical = status.output;
  canonical.append(tracked_diff.output);
  for (const auto & relative : untracked_paths) {
    canonical.append(relative);
    canonical.push_back('\0');
    canonical.append(read_binary_file(std::filesystem::path(path) / relative));
  }

  long ahead = 0, behind = 0;
  std::istringstream divergence_stream(divergence.output);
  divergence_stream >> ahead >> behind;

  return {
    {"path", path},
    {"head", trim_newline(head.output)},
    {"branch", trim_newline(branch.output)},
    {"upstream", upstream.exit_code == 0 ? trim_newline(upstream.output) : ""},
    {"ahead", ahead},
    {"behind", behind},
    {"dirty", !records.empty()},
    {"dirty_files", dirty_files},
    {"untracked_files", untracked_files},
    {"diff_hash_sha256", hash_bytes(canonical)},
    {"diff_hash_scope", "porcelain-v1-z + git-diff-binary-HEAD + untracked-path-and-content"},
  };
}

nlohmann::json reproducibility_metadata(
  const std::string & config_path, const std::string & model_path)
{
  return {
    {"config_path", config_path},
    {"config_hash", file_sha256(config_path)},
    {"model_path", model_path},
    {"model_hash", file_sha256(model_path)},
    {"command", process_command_line()},
    {"timestamp_utc", utc_timestamp()},
    {"repositories",
     {{"ats_vision", git_repository_metadata("/home/kong/vision")},
      {"sp_vision25", git_repository_metadata("/home/kong/vision/sp_vision25")},
      {"bevy_robomaster_simulator",
       git_repository_metadata("/home/kong/vision/bevy_robomaster_simulator")}}},
  };
}
}  // namespace sim_io
