#ifndef TOOLS__PATH_HPP
#define TOOLS__PATH_HPP

#include <filesystem>
#include <initializer_list>
#include <string>

namespace tools
{

// 返回编译期记录的源码根目录。
std::filesystem::path source_root();

// 解析运行时资源路径。
// 适用于 configs/xxx.yaml、assets/xxx.xml 这类原本依赖当前工作目录的相对路径。
std::filesystem::path resolve_runtime_path(const std::filesystem::path & path);

// 解析“同名前缀 + 多个扩展名”的资源路径。
// 适用于 `assets/demo/demo` 这种需要同时找到 `.avi/.txt` 配套文件的场景。
std::filesystem::path resolve_runtime_prefix(
  const std::filesystem::path & path, std::initializer_list<std::string> suffixes);

// 专门用于配置文件路径解析，语义上更明确。
std::filesystem::path resolve_config_path(const std::filesystem::path & path);

// 当 YAML 里的模型路径是相对路径时，优先以配置文件所在位置和包根目录为基准解析。
std::filesystem::path resolve_path_from_config(
  const std::filesystem::path & config_path, const std::filesystem::path & path);

// 便于直接回填给现有 std::string 风格接口。
std::string resolve_runtime_path_string(const std::string & path);
std::string resolve_runtime_prefix_string(
  const std::string & path, std::initializer_list<std::string> suffixes);
std::string resolve_config_path_string(const std::string & path);
std::string resolve_path_from_config_string(
  const std::string & config_path, const std::string & path);

}  // namespace tools

#endif  // TOOLS__PATH_HPP
