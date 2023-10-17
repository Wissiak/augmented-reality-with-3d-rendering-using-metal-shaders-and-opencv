#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;

/*
https://thecodinginterface.com/blog/opencv-cpp-vscode/

mkdir build && cd build
cmake ..
cmake --build . --config Release
*/

class ARWebcam {
private:
  const cv::Mat referenceImage = cv::imread("../book1-reference.png");

  std::vector<cv::KeyPoint> referenceKeypoints;

  cv::Mat referenceDescriptors;
  cv::Ptr<cv::SiftFeatureDetector> detector;
  cv::Ptr<cv::BFMatcher> matcher;
  std::vector<cv::Point2f> corners;
  bool const showMatches = true;

public:
  ARWebcam() {
    detector = cv::SiftFeatureDetector::create(3000, 4, 0.001, 20, 1.5);
    detector->detectAndCompute(referenceImage, cv::noArray(),
                               referenceKeypoints, referenceDescriptors);
    matcher = cv::BFMatcher::create(cv::NORM_L2, true);

    unsigned height = referenceImage.rows;
    unsigned width = referenceImage.cols;
    unsigned data[8] = {0,         0,          width - 1, 0,
                        width - 1, height - 1, 0,         height - 1};

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
      cout << "No video stream detected" << endl;
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
      if (nInliers > 50) {
        std::vector<cv::Point2f> transformedCorners;
        cv::perspectiveTransform(corners, transformedCorners, H);

        for (int i = 0; i < transformedCorners.size(); i++)
          cv::circle(videoFrame, transformedCorners[i], 10, CV_RGB(100, 0, 0),
                     -1, 8, 0);
        // cv::polylines(videoFrame, transformedCorners, true,
        //               cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
      }
      imshow("Video Player", videoFrame);
      char c = (char)cv::waitKey(1);

      // If 'Esc' is entered break the loop
      if (c == 27) {
        break;
      }
    }
    cap.release();
  }
};

int main(int argc, char **argv) {
  std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  ARWebcam arWebcam;
  arWebcam.video_in();

  return 0;
}