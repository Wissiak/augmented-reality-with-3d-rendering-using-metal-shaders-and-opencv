// OpenCV imports must come first
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

class ARWebcam {
private:
  const cv::Mat referenceImage = cv::imread("./assets/book1-reference.png");

  std::vector<cv::KeyPoint> referenceKeypoints;

  cv::Mat referenceDescriptors;
  cv::Ptr<cv::SiftFeatureDetector> detector;
  cv::Ptr<cv::BFMatcher> matcher;
  std::vector<cv::Point2f> corners{4};
  bool const showMatches = false;

  MTLEngine engine;

public:
  ARWebcam(MTLEngine mEngine) {
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
  }

  void video_in() {
    cv::Mat videoFrame;
    cv::namedWindow("Video Player");
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      std::cout << "No video stream detected" << std::endl;
      system("pause");
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
        startPipeline(engine, videoFrame);
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

  void startPipeline(MTLEngine engine, cv::Mat videoFrame) {
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

    texture->getBytes(imageData.data(), bytesPerRow, bytesPerImage, region,
                      level, slice);

    cv::Mat image(texture->height(), texture->width(), CV_8UC4,
                  imageData.data());

    cv::Mat test;
    // Converting the image from BGRA to BGR and saving it in the dst_mat matrix
    cv::cvtColor(videoFrame, test, cv::COLOR_BGR2BGRA);

    std::cout << "Rendered Channels: " << image.channels()
              << ", videoFrame Channels: " << test.channels()
              << std::endl;
    // Overlay images
    cv::Mat dst;
    cv::addWeighted(image, 1, test, 1, 0, dst);

    cv::imshow("Image", dst);
    cv::waitKey(0);
  }
};

int main(int argc, char **argv) {
  std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  MTLEngine engine;
  engine.init();
  ARWebcam arWebcam(engine);

  arWebcam.video_in();

  engine.cleanup();

  return 0;
}