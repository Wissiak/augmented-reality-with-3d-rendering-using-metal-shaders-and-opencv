#include "ARWebcam.hpp"

int main(int argc, char **argv) {
  std::cout << "Current path is " << std::filesystem::current_path() << '\n';

  cv::VideoCapture cap(0);
  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  MTLEngine engine;
  engine.init(width, height);
  ARWebcam arWebcam{engine, cv::Size{width, height}};

  arWebcam.video_in(cap);

  engine.cleanup();

  return 0;
}