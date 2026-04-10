#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"

// 定义命令行参数
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{start-index s  |                        | 视频起始帧下标    }"
  "{end-index e    |                        | 视频结束帧下标    }"
  "{output-path p  | records/Big/2024-05-14_11-6-26     | avi和txt文件的路径}"
  "{@input-path    |                        | avi和txt文件的路径}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto output_path = cli.get<std::string>("output-path");
  auto start_index = cli.get<int>("start-index");
  std::cout << start_index << std::endl;
  auto end_index = cli.get<int>("end-index");

  // 初始化绘图器和退出器
  tools::Exiter exiter;

  // 构造视频和文本文件路径
  // 输入录像一般来自已有文件，因此按“资源路径”方式解析；
  // 输出则显式落到当前工作目录或用户给定的绝对路径，避免误写入安装空间。
  input_path = tools::resolve_runtime_prefix_string(input_path, {".avi", ".txt"});
  const auto output_base_path =
    std::filesystem::path(output_path).is_absolute() ?
    std::filesystem::path(output_path) :
    std::filesystem::absolute(output_path);
  output_path = output_base_path.string();

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);
  if (!video.isOpened() || !text.is_open()) {
    tools::logger()->error("无法打开输入录像或姿态文本: {} / {}", video_path, text_path);
    return 1;
  }

  // 设置视频起始帧
  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }
  // 获取原始视频的参数
  int frameWidth = video.get(cv::CAP_PROP_FRAME_WIDTH);
  int frameHeight = video.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = video.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));

  // 创建输出文件
  std::filesystem::create_directories(output_base_path.parent_path());

  auto outvideo_path = fmt::format("{}.avi", output_path);
  auto outtext_path = fmt::format("{}.txt", output_path);
  cv::VideoWriter outvideo(outvideo_path, fourcc, fps, cv::Size(frameWidth, frameHeight));
  std::ofstream outtext(outtext_path);
  if (!outvideo.isOpened() || !outtext.is_open()) {
    tools::logger()->error("无法创建输出录像或姿态文本: {} / {}", outvideo_path, outtext_path);
    return 1;
  }

  std::string line;
  cv::Mat img;

  // 循环处理视频帧
  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    outvideo.write(img);

    getline(text, line);
    outtext << line << std::endl;

    cv::resize(img, img, cv::Size(img.size().width * 0.8, img.size().height * 0.8));
    cv::imshow("result", img);

    int key = cv::waitKey(1);
    if (key == 'q') break;
  }
  cv::destroyAllWindows();
  text.close();  // 关闭文件
  outtext.close();

  return 0;
}
