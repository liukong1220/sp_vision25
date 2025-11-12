#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
namespace multithread
{

/**
 * @brief 多线程检测器构造函数
 *
 * @param config_path 配置文件路径
 * @param debug 是否为调试模式
 */
MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  // 读取并配置OpenVINO模型
  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 设置模型输入张量信息
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})  // TODO: 输入尺寸应从配置读取
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  // 设置模型预处理步骤
  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  // 编译模型以在指定设备上运行，并针对吞吐量进行优化
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

/**
 * @brief 将图像帧推入队列进行异步检测
 *
 * @param img 待检测的图像
 * @param t 图像帧的时间戳
 */
void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // 计算缩放比例以保持图像宽高比
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // 预处理：创建letterbox图像
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  auto input_port = compiled_model_.input();
  // 创建推理请求并进行异步推理
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  // 将图像、时间戳和推理请求存入队列
  queue_.push({img.clone(), t, std::move(infer_request)});
}

/**
 * @brief 从队列中取出检测结果
 *
 * @return std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> 装甲板列表和时间戳
 */
std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();  // 等待推理完成

  // 后处理
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  // 暂不支持ROI

  return {std::move(armors), t};
}

/**
 * @brief 从队列中取出调试用的检测结果（包含原始图像）
 *
 * @return std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> 原始图像、装甲板列表和时间戳
 */
std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();  // 等待推理完成

  // 后处理
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  // 暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim