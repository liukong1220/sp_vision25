#include "recorder.hpp"

#include <fmt/chrono.h>

#include <filesystem>
#include <string>

#include "math_tools.hpp"
#include "tools/logger.hpp"

namespace tools
{
Recorder::Recorder(double fps, const std::string & output_dir, const std::string & file_prefix)
: init_(false),
  stop_thread_(false),
  fps_(fps),
  output_dir_(output_dir.empty() ? "records" : output_dir),
  file_prefix_(file_prefix),
  queue_(128)
{
  start_time_ = std::chrono::steady_clock::now();
  last_time_ = start_time_;

  std::filesystem::create_directories(output_dir_);
  const auto stem = build_record_stem();
  text_path_ = fmt::format("{}/{}.txt", output_dir_, stem);
  video_path_ = fmt::format("{}/{}.avi", output_dir_, stem);
}

Recorder::~Recorder()
{
  stop_thread_ = true;
  queue_.push({cv::Mat::zeros(0, 0, 0), {0, 0, 0, 0}, std::chrono::steady_clock::now()});
  if (saving_thread_.joinable()) saving_thread_.join();

  if (!init_) return;
  text_writer_.close();
  video_writer_.release();
}

void Recorder::save_to_file()
{
  while (!stop_thread_) {
    FrameData frame;
    queue_.pop(frame);
    if (frame.img.empty()) {
      tools::logger()->debug("Recorder received empty img. Skip this frame.");
      continue;
    }

    video_writer_.write(frame.img);

    const Eigen::Vector4d xyzw = frame.q.coeffs();
    const auto since_begin = tools::delta_time(frame.timestamp, start_time_);
    text_writer_ << fmt::format(
      "{} {} {} {} {}\n", since_begin, xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  }
}

const std::string & Recorder::video_path() const { return video_path_; }

const std::string & Recorder::text_path() const { return text_path_; }

void Recorder::record(
  const cv::Mat & img, const Eigen::Quaterniond & q,
  const std::chrono::steady_clock::time_point & timestamp)
{
  if (img.empty()) return;
  if (!init_) {
    init(img);
    if (!init_) return;
  }

  const auto since_last = tools::delta_time(timestamp, last_time_);
  if (since_last < 1.0 / fps_) return;

  last_time_ = timestamp;
  queue_.push({img, q, timestamp});
}

void Recorder::init(const cv::Mat & img)
{
  text_writer_.open(text_path_);
  const auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  video_writer_ = cv::VideoWriter(video_path_, fourcc, fps_, img.size());
  if (!text_writer_.is_open()) {
    tools::logger()->error("Recorder failed to open text file: {}", text_path_);
    return;
  }
  if (!video_writer_.isOpened()) {
    tools::logger()->error("Recorder failed to open video file: {}", video_path_);
    text_writer_.close();
    return;
  }

  saving_thread_ = std::thread(&Recorder::save_to_file, this);
  init_ = true;
  tools::logger()->info("Recorder saving video to {}", video_path_);
  tools::logger()->info("Recorder saving pose data to {}", text_path_);
}

std::string Recorder::build_record_stem() const
{
  const auto timestamp = fmt::format("{:%Y-%m-%d_%H-%M}", std::chrono::system_clock::now());
  const std::string prefix = file_prefix_.empty() ? "" : file_prefix_ + "_";
  const std::string base_stem = prefix + timestamp;

  std::string stem = base_stem;
  int index = 1;
  while (
    std::filesystem::exists(fmt::format("{}/{}.avi", output_dir_, stem)) ||
    std::filesystem::exists(fmt::format("{}/{}.txt", output_dir_, stem)))
  {
    stem = fmt::format("{}_{:02}", base_stem, index++);
  }
  return stem;
}

}  // namespace tools
