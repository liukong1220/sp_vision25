#include "mt_detector.hpp"

#include <cstring>

#include "tools/openvino_utils.hpp"
#include "tools/path.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{
namespace multithread
{
namespace
{
size_t inference_max_inflight(const std::string & config_path)
{
  const auto yaml = tools::load(config_path);
  return std::max<size_t>(1, tools::read_or<int>(yaml, "inference_max_inflight", 3));
}
}  // namespace

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug),
  max_inflight_(inference_max_inflight(config_path)),
  queue_(
    max_inflight_,
    [] { tools::logger()->debug("[MultiThreadDetector] inference queue is full"); })
{
  auto yaml = tools::load(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = tools::resolve_path_from_config_string(
    config_path, yaml[yolo_name + "_model_path"].as<std::string>());
  device_ = yaml["device"].as<std::string>();

  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})  // TODO
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = tools::ov_utils::compile_model_with_fallback(
    core_, model, device_, "MultiThreadDetector",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info(
    "[MultiThreadDetector] initialized, max in-flight requests: {}", max_inflight_);
}

bool MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // Reject before allocating and starting an InferRequest. This keeps latency bounded
  // when camera throughput is temporarily higher than inference throughput.
  if (queue_.size() >= max_inflight_) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // preproces
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  auto infer_request = compiled_model_.create_infer_request();
  auto input_tensor = infer_request.get_input_tensor();
  std::memcpy(input_tensor.data(), input.data, input.total() * input.elemSize());
  infer_request.start_async();
  const bool queued = queue_.try_push({img.clone(), t, std::move(infer_request)});
  if (queued) {
    submitted_.fetch_add(1, std::memory_order_relaxed);
  } else {
    dropped_.fetch_add(1, std::memory_order_relaxed);
  }
  return queued;
}

size_t MultiThreadDetector::pending() const { return queue_.size(); }

uint64_t MultiThreadDetector::submitted() const
{
  return submitted_.load(std::memory_order_relaxed);
}

uint64_t MultiThreadDetector::dropped() const
{
  return dropped_.load(std::memory_order_relaxed);
}

void MultiThreadDetector::clear() { queue_.clear(); }

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
