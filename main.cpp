// OpenCV imports must come first
#include <Foundation/NSTypes.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
// End OpenCV Imports
#include "MetalEngine.hpp"
#include <filesystem>
#include <opencv2/videoio.hpp>

class ARWebcam {

public:
  ARWebcam(MTLEngine mEngine, cv::Size imgSize) {
    detector = cv::SiftFeatureDetector::create(1000, 4, 0.001, 20, 1.5);
    detector->detectAndCompute(referenceImage, cv::noArray(),
                               referenceKeypoints, referenceDescriptors);
    matcher = cv::BFMatcher::create(cv::NORM_L2, true);

    engine = mEngine;

    unsigned height = referenceImage.rows;
    unsigned width = referenceImage.cols;
    corners.push_back(cv::Point2f(0, 0));
    corners.push_back(cv::Point2f(width - 1, 0));
    corners.push_back(cv::Point2f(width - 1, height - 1));
    corners.push_back(cv::Point2f(0, height - 1));

    bytesPerRow = imgSize.width * 4;
    region.origin = {0, 0, 0};
    region.size = {NS::UInteger(imgSize.width), NS::UInteger(imgSize.height), 1};
  }

  void video_in(cv::VideoCapture cap) {
    cv::Mat videoFrame;
    if (!cap.isOpened()) {
      std::cerr << "No video stream detected" << std::endl;
      return;
    }
    while (true) {
      cap >> videoFrame;
      if (videoFrame.empty()) {
        break;
      }
      std::vector<cv::KeyPoint> frameKeypoints;
      cv::Mat frameDescriptors;
      detector->detectAndCompute(videoFrame, cv::noArray(), frameKeypoints,
                                 frameDescriptors);

      std::vector<cv::DMatch> matches;
      matcher->match(referenceDescriptors, frameDescriptors, matches,
                     cv::noArray());

      std::vector<cv::Point2f> obj;
      std::vector<cv::Point2f> scene;
      for (size_t i = 0; i < matches.size(); i++) {
        obj.push_back(referenceKeypoints[matches[i].queryIdx].pt);
        scene.push_back(frameKeypoints[matches[i].trainIdx].pt);
      }

      cv::Mat mask;
      cv::Mat H = findHomography(obj, scene, cv::RANSAC, 5.0, mask);

      int nInliers = cv::countNonZero(mask);
      std::cout << "Inliers: " << nInliers << std::endl;

      if (showMatches) {
        cv::Mat matchFrame;
        cv::copyTo(videoFrame, matchFrame, cv::noArray());
        // matches.erase(matches.begin() + 400, matches.end());
        cv::drawMatches(referenceImage, referenceKeypoints, videoFrame,
                        frameKeypoints, matches, matchFrame, 2,
                        cv::Scalar::all(-1), NULL, mask,
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Reliable Matches", matchFrame);
      }
      if (nInliers > 80) {
        std::vector<cv::Point2f> transformedCorners;
        cv::perspectiveTransform(corners, transformedCorners, H);

        for (int i = 0; i < transformedCorners.size(); i++) {
          cv::line(videoFrame, transformedCorners[i],
                   transformedCorners[(i + 1) % transformedCorners.size()],
                   CV_RGB(0, 255, 0), 3, 8, 0);
        }
      }
      // imshow("Video Player", videoFrame);
      try {
        float3 position = make_float3(20, 40, 20);
        float yaw = rand() % 20 + -110;
        float pitch = rand() % 20 + -30;
        startPipeline(engine, videoFrame, position, pitch, yaw);
      } catch (const std::exception &ex) {
        std::cerr << "Render Pipeline error: " << ex.what() << std::endl;
      }
      char c = (char)cv::waitKey(1);

      // End loop on ESC
      if (c == 27) {
        break;
      }
    }
    cap.release();
  }

private:
  const cv::Mat referenceImage = cv::imread("./assets/book1-reference.png");

  std::vector<cv::KeyPoint> referenceKeypoints;

  cv::Mat referenceDescriptors;
  cv::Ptr<cv::SiftFeatureDetector> detector;
  cv::Ptr<cv::BFMatcher> matcher;
  std::vector<cv::Point2f> corners{4};
  bool const showMatches = false;

  MTLEngine engine;

  MTL::Region region;
  NS::UInteger bytesPerRow;
  NS::UInteger bytesPerImage = 0;
  NS::UInteger level = 0;
  NS::UInteger slice = 0;

  void startPipeline(MTLEngine engine, cv::Mat videoFrame, float3 position,
                     float pitch, float yaw) {
    CA::MetalDrawable *drawable = engine.run(position, pitch, yaw);
    auto texture = drawable->texture();

    std::vector<uint8_t> imageData(texture->width() * texture->height() * 4);

    texture->getBytes(imageData.data(), bytesPerRow, bytesPerImage, region,
                      level, slice);

    cv::Mat image(texture->height(), texture->width(), CV_8UC4,
                  imageData.data());

    // Converting the image from BGR to BGRA in order to overlay the images
    cv::cvtColor(videoFrame, videoFrame, cv::COLOR_BGR2BGRA);

    // Overlay images
    cv::Mat dst;
    cv::addWeighted(image, 1, videoFrame, 1, 0, dst);

    cv::imshow("AR Video Output", dst);
  }
};

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