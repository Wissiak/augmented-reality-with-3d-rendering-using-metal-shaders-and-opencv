#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "MetalEngine.hpp"
#include <filesystem>

void startPipeline(MTLEngine engine) {
  CA::MetalDrawable *drawable = engine.run();
  auto texture = drawable->texture();
  MTL::Region region;
  region.origin = {0, 0, 0};
  region.size = {texture->width(), texture->height(), 1};
  NS::UInteger bytesPerRow = texture->width() * 4; // Assuming RGBA format
  NS::UInteger bytesPerImage = 0;                  // For 2D textures
  NS::UInteger level = 0;                          // Mipmap level
  NS::UInteger slice = 0;

  std::vector<uint8_t> imageData(texture->width() * texture->height() * 4);

  texture->getBytes(imageData.data(), bytesPerRow, bytesPerImage, region, level,
                    slice);

  cv::Mat image(texture->height(), texture->width(), CV_8UC4, imageData.data());
  cv::imshow("Image", image);
  cv::waitKey(0);
}

int main(int argc, char *argv[]) {
  std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  MTLEngine engine;
  engine.init();

  startPipeline(engine);

  engine.cleanup();
}
