#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include <limits>
#include <vector>

const std::string keys =
  "{help h usage ? |                          | print help message }"
  "{config-path c  | configs/calibration.yaml | path to calibration yaml }"
  "{@input-folder  | assets/img_with_q        | folder containing captured images }";

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, float center_distance)
{
  std::vector<cv::Point3f> points;
  points.reserve(pattern_size.width * pattern_size.height);

  for (int i = 0; i < pattern_size.height; i++) {
    for (int j = 0; j < pattern_size.width; j++) {
      points.push_back({j * center_distance, i * center_distance, 0});
    }
  }

  return points;
}

void load(
  const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
  std::vector<std::vector<cv::Point3f>> & obj_points,
  std::vector<std::vector<cv::Point2f>> & img_points)
{
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  for (int i = 1; true; i++) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;

    img_size = img.size();

    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findCirclesGrid(
      img, pattern_size, centers_2d, cv::CALIB_CB_SYMMETRIC_GRID);

    auto drawing = img.clone();
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);
    cv::imshow("Press any to continue", drawing);
    cv::waitKey(0);

    fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
    if (!success) continue;

    img_points.emplace_back(centers_2d);
    obj_points.emplace_back(centers_3d(pattern_size, static_cast<float>(center_distance_mm)));
  }
}

void print_yaml(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double error)
{
  YAML::Emitter result;
  std::vector<double> camera_matrix_data(
    camera_matrix.begin<double>(), camera_matrix.end<double>());
  std::vector<double> distort_coeffs_data(
    distort_coeffs.begin<double>(), distort_coeffs.end<double>());

  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("reprojection error {:.4f}px", error));
  result << YAML::Key << "camera_matrix";
  result << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs";
  result << YAML::Value << YAML::Flow << distort_coeffs_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  cv::Size img_size;
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> img_points;
  load(input_folder, config_path, img_size, obj_points, img_points);

  if (obj_points.size() < 5) {
    fmt::print(
      "Calibration aborted: only {} valid images found, at least 5 are required.\n",
      obj_points.size());
    return 1;
  }

  fmt::print("Valid image count: {}\n", obj_points.size());

  cv::Mat camera_matrix, distort_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  auto criteria = cv::TermCriteria(
    cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, std::numeric_limits<double>::epsilon());
  cv::calibrateCamera(
    obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs, cv::CALIB_FIX_K3,
    criteria);

  double error_sum = 0;
  size_t total_points = 0;
  for (size_t i = 0; i < obj_points.size(); i++) {
    std::vector<cv::Point2f> reprojected_points;
    cv::projectPoints(
      obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);

    total_points += reprojected_points.size();
    for (size_t j = 0; j < reprojected_points.size(); j++) {
      error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
    }
  }

  if (total_points == 0) {
    fmt::print("Calibration aborted: zero valid reprojected points.\n");
    return 1;
  }

  auto error = error_sum / total_points;
  print_yaml(camera_matrix, distort_coeffs, error);
  return 0;
}
