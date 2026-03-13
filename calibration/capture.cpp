#include <fmt/core.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ?  |                          | print help message }"
  "{config-path c   | configs/calibration.yaml | path to calibration yaml }"
  "{output-folder o | assets/img_with_q        | output folder for img and q }";

void write_q(const std::string & q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // Save quaternion in wxyz order.
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(
  const std::string & config_path, const std::string & output_folder, const cv::Size & pattern_size)
{
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = gimbal.q(timestamp);

    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Yaw {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Pitch {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Roll {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findCirclesGrid(
      img, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);
    cv::drawChessboardCorners(img_with_ypr, pattern_size, centers_2d, success);
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);

    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    if (!success) {
      tools::logger()->warn("Pattern not found, sample not saved.");
      continue;
    }

    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");
  auto output_folder = cli.get<std::string>("output-folder");

  auto yaml = tools::load(config_path);
  auto pattern_cols = tools::read<int>(yaml, "pattern_cols");
  auto pattern_rows = tools::read<int>(yaml, "pattern_rows");
  cv::Size pattern_size(pattern_cols, pattern_rows);

  std::filesystem::create_directories(output_folder);

  tools::logger()->info("Pattern size: {} x {}", pattern_cols, pattern_rows);
  capture_loop(config_path, output_folder, pattern_size);

  tools::logger()->warn("Quaternion output order is wxyz");

  return 0;
}
