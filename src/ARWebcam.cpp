#include "ARWebcam.hpp"
#include "opencv2/core.hpp"
#include <simd/vector_make.h>

auto ARWebcam::findPoseTransformationParamsEngineeringMethod(
    const cv::Size &shape, const std::vector<cv::Point2f> &x_d,
    const std::vector<cv::Point2f> &x_u, cv::Mat &R_c_b, cv::Mat &t_c_cb) -> bool {

  cv::Point2f x_d_center(shape.width / 2.0, shape.height / 2.0);

  std::vector<double> ratio;
  std::vector<cv::Mat> solution;
  std::vector<double> angle;

  for (int iter = 0; iter < rotations * steps_per_rotation; ++iter) {
    double alpha = iter * 2 * CV_PI / steps_per_rotation;
    double dr =
        static_cast<double>(iter) / steps_per_rotation * delta_per_rotation;
    double dx = dr * cos(alpha);
    double dy = dr * sin(alpha);
    std::vector<cv::Point2f> x_ds = x_d;
    for (cv::Point2f &point : x_ds) {
      point.x += dx;
      point.y += dy;
    }
    cv::Mat tmp;
    std::vector<cv::Point2f> sub(4, x_d_center);
    cv::subtract(x_ds, sub, tmp);
    cv::Mat cH_c_b = homographyFrom4PointCorrespondences(tmp, x_u);

    cv::Mat R_c_b, t_c_cb;
    double fx, fy;
    // Call your recoverRigidBodyMotionAndFocalLengths function
    recoverRigidBodyMotionAndFocalLengths(cH_c_b, R_c_b, t_c_cb, fx, fy);

    if (!std::isnan(fx)) {
      ratio.push_back(std::min(fx, fy) / std::max(fx, fy));
      angle.push_back(alpha);
      solution.push_back(R_c_b);
      solution.push_back(t_c_cb);
    }
  }

  if (ratio.empty()) {
    std::cout << "Could not find a Rotation Matrix and Transformation Vector"
              << std::endl;
    return false;
  }

  // Identify the most plausible solution
  double maxRatio = ratio[0];
  int idx = 0;
  for (size_t i = 1; i < ratio.size(); ++i) {
    if (ratio[i] > maxRatio) {
      maxRatio = ratio[i];
      idx = i;
    }
  }

  R_c_b = solution[idx * 3];      // Extract R_c_b
  t_c_cb = solution[idx * 3 + 1]; // Extract t_c_cb
  return true;
}

auto ARWebcam::recoverRigidBodyMotionAndFocalLengths(const cv::Mat &H_c_b,
                                                     cv::Mat &R_c_b,
                                                     cv::Mat &t_c_cb,
                                                     double &fx, double &fy)
    -> void {
  cv::Mat a = H_c_b;

  cv::Mat Ma =
      (cv::Mat_<double>(3, 3) << a.at<double>(0, 0) * a.at<double>(0, 0),
       a.at<double>(1, 0) * a.at<double>(1, 0),
       a.at<double>(2, 0) * a.at<double>(2, 0),
       a.at<double>(0, 1) * a.at<double>(0, 1),
       a.at<double>(1, 1) * a.at<double>(1, 1),
       a.at<double>(2, 1) * a.at<double>(2, 1),
       a.at<double>(0, 0) * a.at<double>(0, 1),
       a.at<double>(1, 0) * a.at<double>(1, 1),
       a.at<double>(2, 0) * a.at<double>(2, 1));

  cv::Mat y = (cv::Mat_<double>(3, 1) << 1, 1, 0);
  cv::Mat diags =cv::Mat::diag( Ma.inv() * y);

  std::cout << diags << std::endl;

  cv::sqrt(diags, diags);

  cv::Mat B = diags * H_c_b;
  cv::Mat rx = B.col(0);
  cv::Mat ry = B.col(1);
  cv::Mat rz = rx.cross(ry);
  R_c_b = cv::Mat(3, 3, CV_64F);
  rx.copyTo(R_c_b.col(0));
  ry.copyTo(R_c_b.col(1));
  rz.copyTo(R_c_b.col(2));

  t_c_cb = B.col(2);
  fx = diags.at<double>(2, 2) / diags.at<double>(0, 0);
  fy = diags.at<double>(2, 2) / diags.at<double>(1, 1);
}

ARWebcam::ARWebcam(MTLEngine mEngine, cv::Size imgSize) {
  detector = cv::SiftFeatureDetector::create(3000, 8, 0.001, 20, 1.5);
  detector->detectAndCompute(referenceImage, cv::noArray(), referenceKeypoints,
                             referenceDescriptors);
  matcher = cv::BFMatcher::create(cv::NORM_L2, true);

  engine = mEngine;

  unsigned height = referenceImage.rows;
  unsigned width = referenceImage.cols;
  corners = std::vector{cv::Point2f(0, 0), cv::Point2f(width - 1, 0),
                        cv::Point2f(width - 1, height - 1),
                        cv::Point2f(0, height - 1)};

  bytesPerRow = imgSize.width * 4;
  region.origin = {0, 0, 0};
  region.size = {NS::UInteger(imgSize.width), NS::UInteger(imgSize.height), 1};

  this->imgSize = imgSize;
}

auto ARWebcam::video_in(cv::VideoCapture cap) -> void {
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
      cv::Mat R_c_b;
      cv::Mat t_c_cb;
      auto success = findPoseTransformationParamsNew(
          imgSize, transformedCorners, corners, R_c_b, t_c_cb);
      if (success) {
        try {
          startPipeline(videoFrame, R_c_b, t_c_cb);
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

auto ARWebcam::startPipeline(cv::Mat &videoFrame, cv::Mat &R_c_b,
                             cv::Mat &t_c_cb) -> void {
  cv::Mat rotationAxis;
  cv::Rodrigues(R_c_b, rotationAxis);
  double theta = cv::norm(rotationAxis);
  rotationAxis = rotationAxis / theta;
  t_c_cb = t_c_cb / scalingFactor;

  double x = t_c_cb.at<double>(0);
  double y = t_c_cb.at<double>(1);
  double z = t_c_cb.at<double>(2);
  float3 t = make_float3(x, y, z);
  auto translationMatrix = matrix4x4_translation(t.xyz);
  matrix_float4x4 rotationMatrix = matrix4x4_rotation(
      theta, rotationAxis.at<double>(0), rotationAxis.at<double>(1),
      rotationAxis.at<double>(2));
  matrix_float4x4 modelMatrix = translationMatrix * rotationMatrix;

  float3 viewDir = normalize(t);
  float3 factor = t.xyz / viewDir.xyz;
  factor.x -= 100;
  factor.y += 100;
  factor.z -= 20;

  simd_float4 lightPosition = simd_make_float4(
      viewDir.x * factor.x, viewDir.y * factor.y, viewDir.z * factor.z, 1);

  CA::MetalDrawable *drawable =
      engine.run(lightPosition, pitch, yaw, modelMatrix);
  auto texture = drawable->texture();

  std::vector<uint8_t> imageData(imgSize.width * imgSize.height * 4);

  texture->getBytes(imageData.data(), bytesPerRow, bytesPerImage, region, level,
                    slice);

  cv::Mat image(imgSize.height, imgSize.width, CV_8UC4, imageData.data());

  // Converting the image from BGR to BGRA in order to overlay the images
  cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

  // Overlay images
  cv::copyTo(image, videoFrame, image);

  cv::imshow("AR Video Output", videoFrame);
}

auto ARWebcam::focalLength(const cv::Mat &H_c_b) -> double {
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

auto ARWebcam::rigidBodyMotion(const cv::Mat &H_c_b, double f, cv::Mat &R_c_b,
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

  t_c_cb = V.col(2);
}

auto ARWebcam::homographyFrom4PointCorrespondences(
    const std::vector<cv::Point2f> &x_d, const std::vector<cv::Point2f> &x_u)
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

auto ARWebcam::findPoseTransformationParamsNew(
    const cv::Size &shape, const std::vector<cv::Point2f> &x_d,
    const std::vector<cv::Point2f> &x_u, cv::Mat &R_c_b, cv::Mat &t_c_cb) -> bool {
  cv::Point2f x_d_center(shape.width / 2.0, shape.height / 2.0);

  // Assuming you have a function homographyFrom4PointCorrespondences already
  // implemented
  std::vector<cv::Point2f> sub(4, x_d_center);
  cv::subtract(x_d, sub, x_d);
  cv::Mat cH_c_b = homographyFrom4PointCorrespondences(x_d, x_u);

  cv::Mat nanMask = cv::Mat(cH_c_b != cH_c_b);
  if (cv::countNonZero(nanMask) > 0) {
    return false;
  }

  double f = focalLength(cH_c_b);

  try {
    rigidBodyMotion(cH_c_b, f, R_c_b, t_c_cb);
  } catch (...) {
    std::cout << "Could not resolve pose" << std::endl;
    return false;
  }

  return true;
}