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
#include "render/MetalEngine.hpp"
#include <filesystem>
#include <opencv2/videoio.hpp>

class ARWebcam {
public:
  ARWebcam(MTLEngine mEngine, cv::Size imgSize);

  auto video_in(cv::VideoCapture cap) -> void;

private:
  const cv::Mat referenceImage = cv::imread("./assets/book1-reference.png");

  std::vector<cv::KeyPoint> referenceKeypoints;

  cv::Mat referenceDescriptors;
  cv::Ptr<cv::SiftFeatureDetector> detector;
  cv::Ptr<cv::BFMatcher> matcher;
  std::vector<cv::Point2f> corners;
  bool const showMatches = false;
  int const scalingFactor = 5;
  float const yaw = 90;
  float const pitch = 0;
  cv::Size imgSize;

  MTLEngine engine;

  int const rotations = 3;
  int const steps_per_rotation = 50;
  int const delta_per_rotation = 6;

  MTL::Region region;
  NS::UInteger bytesPerRow;
  NS::UInteger bytesPerImage = 0;
  NS::UInteger level = 0;
  NS::UInteger slice = 0;

  auto startPipeline(cv::Mat &videoFrame, cv::Mat &R_c_b, cv::Mat &t_c_cb)
      -> void;

  auto focalLength(const cv::Mat &H_c_b) -> double;

  auto rigidBodyMotion(const cv::Mat &H_c_b, double f, cv::Mat &R_c_b,
                       cv::Mat &t_c_cb) -> void;

  auto homographyFrom4PointCorrespondences(const std::vector<cv::Point2f> &x_d,
                                           const std::vector<cv::Point2f> &x_u)
      -> cv::Mat;

  auto findPoseTransformationParamsNew(const cv::Size &shape,
                                       const std::vector<cv::Point2f> &x_d,
                                       const std::vector<cv::Point2f> &x_u,
                                       cv::Mat &R_c_b, cv::Mat &t_c_cb) -> bool;

  auto recoverRigidBodyMotionAndFocalLengths(const cv::Mat &H_c_b,
                                             cv::Mat &R_c_b, cv::Mat &t_c_cb,
                                             double &fx, double &fy) -> void;

  auto findPoseTransformationParamsEngineeringMethod(const cv::Size &shape,
                                    const std::vector<cv::Point2f> &x_d,
                                    const std::vector<cv::Point2f> &x_u,
                                    cv::Mat &R_c_b, cv::Mat &t_c_cb) -> bool;
};