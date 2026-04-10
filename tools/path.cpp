#include "tools/path.hpp"

#include <unistd.h>

#include <array>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace tools
{
namespace
{

fs::path weakly_normalize(const fs::path & path)
{
  std::error_code ec;
  auto normalized = fs::weakly_canonical(path, ec);
  return ec ? path : normalized;
}

bool path_exists(const fs::path & path)
{
  std::error_code ec;
  return fs::exists(path, ec);
}

bool prefix_exists(const fs::path & path, std::initializer_list<std::string> suffixes)
{
  if (path_exists(path)) return true;
  if (suffixes.size() == 0) return false;

  for (const auto & suffix : suffixes) {
    auto companion = path;
    companion += suffix;
    if (!path_exists(companion)) return false;
  }

  return true;
}

fs::path executable_path()
{
  std::array<char, 4096> buffer{};
  const auto length = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) return {};
  buffer[static_cast<std::size_t>(length)] = '\0';
  return fs::path(buffer.data());
}

std::vector<fs::path> candidate_roots()
{
  std::vector<fs::path> roots;

  // 先尝试当前工作目录，兼容用户仍在源码根目录下直接执行旧用法。
  roots.push_back(fs::current_path());

  // 再尝试编译时记录的源码根目录，兼容“在任意目录启动 build 目录中的可执行文件”。
  roots.push_back(source_root());

  // 如果可执行文件来自 `install/<pkg>/lib/<pkg>/xxx`，
  // 则这里可以反推出 share/<pkg> 目录。
  const auto exe = executable_path();
  if (!exe.empty()) {
    const auto prefix = exe.parent_path().parent_path().parent_path();
    roots.push_back(prefix / "share" / "sp_vision25");
  }

  return roots;
}

}  // namespace

fs::path source_root()
{
  return fs::path(SP_VISION_SOURCE_DIR);
}

fs::path resolve_runtime_path(const fs::path & path)
{
  if (path.empty()) return path;
  if (path.is_absolute()) return path;
  if (path_exists(path)) return weakly_normalize(path);

  for (const auto & root : candidate_roots()) {
    if (root.empty()) continue;
    const auto candidate = root / path;
    if (path_exists(candidate)) return weakly_normalize(candidate);
  }

  return path;
}

fs::path resolve_runtime_prefix(const fs::path & path, std::initializer_list<std::string> suffixes)
{
  if (path.empty()) return path;
  if (path.is_absolute()) return path;
  if (prefix_exists(path, suffixes)) return weakly_normalize(path);

  for (const auto & root : candidate_roots()) {
    if (root.empty()) continue;
    const auto candidate = root / path;
    if (prefix_exists(candidate, suffixes)) return weakly_normalize(candidate);
  }

  return path;
}

fs::path resolve_config_path(const fs::path & path)
{
  return resolve_runtime_path(path);
}

fs::path resolve_path_from_config(const fs::path & config_path, const fs::path & path)
{
  if (path.empty()) return path;
  if (path.is_absolute()) return path;
  if (path_exists(path)) return weakly_normalize(path);

  const auto resolved_config = resolve_config_path(config_path);
  if (!resolved_config.empty()) {
    const auto config_dir = resolved_config.parent_path();

    const auto beside_config = config_dir / path;
    if (path_exists(beside_config)) return weakly_normalize(beside_config);

    // 当前项目的模型路径通常写成 `assets/...`，而配置放在 `configs/...`。
    // 所以这里额外尝试一次“配置目录的父目录”。
    const auto package_root_candidate = config_dir.parent_path() / path;
    if (path_exists(package_root_candidate)) return weakly_normalize(package_root_candidate);
  }

  return resolve_runtime_path(path);
}

std::string resolve_runtime_path_string(const std::string & path)
{
  return resolve_runtime_path(fs::path(path)).string();
}

std::string resolve_runtime_prefix_string(
  const std::string & path, std::initializer_list<std::string> suffixes)
{
  return resolve_runtime_prefix(fs::path(path), suffixes).string();
}

std::string resolve_config_path_string(const std::string & path)
{
  return resolve_config_path(fs::path(path)).string();
}

std::string resolve_path_from_config_string(
  const std::string & config_path, const std::string & path)
{
  return resolve_path_from_config(fs::path(config_path), fs::path(path)).string();
}

}  // namespace tools
