// OpenCV imports must come first
#include <Foundation/NSTypes.hpp>
#include <opencv2/calib3d.hpp>
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
    detector = cv::SiftFeatureDetector::create(3000, 8, 0.001, 20, 1.5);
    detector->detectAndCompute(referenceImage, cv::noArray(),
                               referenceKeypoints, referenceDescriptors);
    matcher = cv::BFMatcher::create(cv::NORM_L2, true);

engine = mEngine;

    unsigned height = referenceImage.rows;
    unsigned width = referenceImage.cols;
    corners = std::vector{cv::Point2f(0, 0), cv::Point2f(width - 1, 0),
                          cv::Point2f(width - 1, height - 1),
                          cv::Point2f(0, height - 1)};

    bytesPerRow = imgSize.width * 4;
    region.origin = {0, 0, 0};
    region.size = {NS::UInteger(imgSize.width), NS::UInteger(imgSize.height),
                   1};

    this->imgSize = imgSize;
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
        cv::Mat K_c;
        cv::Mat R_c_b;
        cv::Mat t_c_cb;
        auto success = findPoseTransformationParamsNew(
            imgSize, transformedCorners, corners, R_c_b, t_c_cb, K_c);
        if (success) {
          try {
            startPipeline(engine, videoFrame, R_c_b, t_c_cb);
          } catch (const std::exception &ex) {
            std::cerr << "Render Pipeline error: " << ex.what() << std::endl;
          }
        } else {
          imshow("AR Video Output", videoFrame);
        }
      } else {
        imshow("AR Video Output", videoFrame);
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
  const cv::Mat referenceImage = cv::imread("./assets/private/bfs-reference.jpg");

  std::vector<cv::KeyPoint> referenceKeypoints;

  cv::Mat referenceDescriptors;
  cv::Ptr<cv::SiftFeatureDetector> detector;
  cv::Ptr<cv::BFMatcher> matcher;
  std::vector<cv::Point2f> corners;
  bool const showMatches = false;
  int const scalingFactor = 10;
  cv::Size imgSize;

  MTLEngine engine;

  MTL::Region region;
  NS::UInteger bytesPerRow;
  NS::UInteger bytesPerImage = 0;
  NS::UInteger level = 0;
  NS::UInteger slice = 0;

  void startPipeline(MTLEngine engine, cv::Mat videoFrame, cv::Mat R_c_b,
                     cv::Mat t_c_cb) {
    float3 position = make_float3(0, 0, 0);
    float yaw = 90;
    float pitch = 0;

    cv::Mat rotationAxis;
    cv::Rodrigues(R_c_b, rotationAxis);
    double theta = cv::norm(rotationAxis);
    rotationAxis = -1 * rotationAxis / theta;
    t_c_cb = t_c_cb / scalingFactor;
    matrix_float4x4 rotationMatrix = matrix4x4_rotation(
        theta, rotationAxis.at<double>(0), rotationAxis.at<double>(1),
        rotationAxis.at<double>(2));
    matrix_float4x4 modelMatrix =
        matrix4x4_translation(t_c_cb.at<double>(0), t_c_cb.at<double>(1),
                              t_c_cb.at<double>(2)) *
        rotationMatrix;

    CA::MetalDrawable *drawable =
        engine.run(position, pitch, yaw, modelMatrix);
    auto texture = drawable->texture();

    std::vector<uint8_t> imageData(imgSize.width * imgSize.height * 4);

    texture->getBytes(imageData.data(), bytesPerRow, bytesPerImage, region,
                      level, slice);

    cv::Mat image(imgSize.height, imgSize.width, CV_8UC4, imageData.data());

    // Converting the image from BGR to BGRA in order to overlay the images
    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

    // Overlay images
    cv::Mat dst;
    cv::addWeighted(videoFrame, 1, image, 1, 0, dst);

    cv::imshow("AR Video Output", dst);
  }

  auto focalLength(const cv::Mat &H_c_b) -> double {
    double h11 = H_c_b.at<double>(0, 0);
    double h12 = H_c_b.at<double>(0, 1);
    double h21 = H_c_b.at<double>(1, 0);
    double h22 = H_c_b.at<double>(1, 1);
    double h31 = H_c_b.at<double>(2, 0); // Contains f^2
    double h32 = H_c_b.at<double>(2, 1); // Contains f^2

    // First two columns are orthogonal: 0 = h11*h12 + h21*h22 + h31*h32
    // Solve for f:
    double fsquare = -(h11 * h12 + h21 * h22) / (h31 * h32);

    return std::sqrt(fsquare);
  };

  auto rigidBodyMotion(const cv::Mat &H_c_b, double f, cv::Mat &R_c_b,
                       cv::Mat &t_c_cb) -> void {
    cv::Mat K_c = (cv::Mat_<double>(3, 3) << f, 0, 0, 0, f, 0, 0, 0, 1);

    cv::Mat V = K_c.inv() * H_c_b;
    V = V / cv::norm(V.col(0));

    cv::Mat rx = V.col(0);
    cv::Mat ry = V.col(1) / cv::norm(V.col(1));

    cv::Mat rz = rx.cross(ry);

    R_c_b = cv::Mat(3, 3, CV_64F);
    cv::hconcat(rx, ry, R_c_b);
    cv::hconcat(R_c_b, rz, R_c_b);
    R_c_b = R_c_b.t();

    t_c_cb = V.col(2);
  }

  auto homographyFrom4PointCorrespondences(const std::vector<cv::Point2f> &x_d,
                                           const std::vector<cv::Point2f> &x_u)
      -> cv::Mat {
    cv::Mat A(8, 8, CV_64F);
    cv::Mat y(8, 1, CV_64F);

    for (int n = 0; n < 4; n++) {
      A.at<double>(2 * n, 0) = x_u[n].x;
      A.at<double>(2 * n, 1) = x_u[n].y;
      A.at<double>(2 * n, 2) = 1;
      A.at<double>(2 * n, 3) = 0;
      A.at<double>(2 * n, 4) = 0;
      A.at<double>(2 * n, 5) = 0;
      A.at<double>(2 * n, 6) = -x_u[n].x * x_d[n].x;
      A.at<double>(2 * n, 7) = -x_u[n].y * x_d[n].x;

      A.at<double>(2 * n + 1, 0) = 0;
      A.at<double>(2 * n + 1, 1) = 0;
      A.at<double>(2 * n + 1, 2) = 0;
      A.at<double>(2 * n + 1, 3) = x_u[n].x;
      A.at<double>(2 * n + 1, 4) = x_u[n].y;
      A.at<double>(2 * n + 1, 5) = 1;
      A.at<double>(2 * n + 1, 6) = -x_u[n].x * x_d[n].y;
      A.at<double>(2 * n + 1, 7) = -x_u[n].y * x_d[n].y;

      y.at<double>(2 * n, 0) = x_d[n].x;
      y.at<double>(2 * n + 1, 0) = x_d[n].y;
    }

    // Compute coefficient vector theta = [a, b, c, ... , h]
    cv::Mat theta;
    cv::solve(A, y, theta);

    // Create the homography matrix
    cv::Mat H_d_u(3, 3, CV_64F);
    H_d_u.at<double>(0, 0) = theta.at<double>(0);
    H_d_u.at<double>(0, 1) = theta.at<double>(1);
    H_d_u.at<double>(0, 2) = theta.at<double>(2);
    H_d_u.at<double>(1, 0) = theta.at<double>(3);
    H_d_u.at<double>(1, 1) = theta.at<double>(4);
    H_d_u.at<double>(1, 2) = theta.at<double>(5);
    H_d_u.at<double>(2, 0) = theta.at<double>(6);
    H_d_u.at<double>(2, 1) = theta.at<double>(7);
    H_d_u.at<double>(2, 2) = 1;

    return H_d_u;
  }

  auto findPoseTransformationParamsNew(const cv::Size &shape,
                                       const std::vector<cv::Point2f> &x_d,
                                       const std::vector<cv::Point2f> &x_u,
                                       cv::Mat &R_c_b, cv::Mat &t_c_cb,
                                       cv::Mat &K_c) -> bool {
    cv::Point2f x_d_center(shape.width / 2.0, shape.height / 2.0);

    // Assuming you have a function homographyFrom4PointCorrespondences already
    // implemented
    std::vector<cv::Point2f> sub(4, x_d_center);
    cv::subtract(x_d, sub, x_d);
    cv::Mat cH_c_b = homographyFrom4PointCorrespondences(x_d, x_u);

    double f = focalLength(cH_c_b);

    try {
      rigidBodyMotion(cH_c_b, f, R_c_b, t_c_cb);
    } catch (...) {
      std::cout << "Could not resolve pose" << std::endl;
      return false;
    }

    K_c = (cv::Mat_<double>(3, 3) << f, 0, x_d_center.x, 0, f, x_d_center.y, 0,
           0, 1);
    return true;
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