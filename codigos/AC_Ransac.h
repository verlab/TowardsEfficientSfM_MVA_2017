// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define OPENMVG_USE_OPENMP

#include "openMVG/sfm/pipelines/sfm_robust_model_estimation.hpp"

#include "openMVG/multiview/solver_essential_kernel.hpp"
#include "openMVG/multiview/projection.hpp"
#include "openMVG/multiview/triangulation.hpp"

#include "openMVG/multiview/solver_resection_kernel.hpp"
#include "openMVG/multiview/solver_resection_p3p.hpp"
#include "openMVG/multiview/solver_essential_kernel.hpp"
#include "openMVG/multiview/projection.hpp"

#include "openMVG/robust_estimation/robust_estimator_ACRansac.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp"

namespace openMVG {
namespace sfm {

bool estimate_Rt_fromE(const Mat3 & K1, const Mat3 & K2,
  const Mat & x1, const Mat & x2,
  const Mat3 & E, const std::vector<size_t> & vec_inliers,
  Mat3 * R, Vec3 * t)
{
  // Accumulator to find the best solution
  std::vector<size_t> f(4, 0);

  std::vector<Mat3> Es; // Essential,
  std::vector<Mat3> Rs;  // Rotation matrix.
  std::vector<Vec3> ts;  // Translation matrix.

  Es.push_back(E);
  // Recover best rotation and translation from E.
  MotionFromEssential(E, &Rs, &ts);

  //-> Test the 4 solutions will all the point
  assert(Rs.size() == 4);
  assert(ts.size() == 4);

  Mat34 P1, P2;
  Mat3 R1 = Mat3::Identity();
  Vec3 t1 = Vec3::Zero();
  P_From_KRt(K1, R1, t1, &P1);

  for (unsigned int i = 0; i < 4; ++i)
  {
    const Mat3 &R2 = Rs[i];
    const Vec3 &t2 = ts[i];
    P_From_KRt(K2, R2, t2, &P2);
    Vec3 X;

    for (size_t k = 0; k < vec_inliers.size(); ++k)
    {
      const Vec2 & x1_ = x1.col(vec_inliers[k]),
        &x2_ = x2.col(vec_inliers[k]);
      TriangulateDLT(P1, x1_, P2, x2_, &X);
      // Test if point is front to the two cameras.
      if (Depth(R1, t1, X) > 0 && Depth(R2, t2, X) > 0)
      {
        ++f[i];
      }
    }
  }
  // Check the solution:
  const std::vector<size_t>::iterator iter = max_element(f.begin(), f.end());
  if (*iter == 0)
  {
    std::cerr << std::endl << "/!\\There is no right solution,"
      << " probably intermediate results are not correct or no points"
      << " in front of both cameras" << std::endl;
    return false;
  }
  const size_t index = std::distance(f.begin(), iter);
  (*R) = Rs[index];
  (*t) = ts[index];

  return true;
}

using namespace openMVG::robust;

bool robustRelativePose(
  const Mat3 & K1, const Mat3 & K2,
  const Mat & x1, const Mat & x2,
  RelativePose_Info & relativePose_info,
  const std::pair<size_t, size_t> & size_ima1,
  const std::pair<size_t, size_t> & size_ima2,
  const size_t max_iteration_count)
{
  // Use the 5 point solver to estimate E
  typedef openMVG::essential::kernel::FivePointKernel SolverType;
  // Define the AContrario adaptor
  typedef ACKernelAdaptorEssential<
      SolverType,
      openMVG::fundamental::kernel::EpipolarDistanceError,
      UnnormalizerT,
      Mat3>
      KernelType;

  KernelType kernel(x1, size_ima1.first, size_ima1.second,
                    x2, size_ima2.first, size_ima2.second, K1, K2);

  // Robustly estimation of the Essential matrix and it's precision
  std::pair<double,double> acRansacOut = ACRANSAC(kernel, relativePose_info.vec_inliers,
    max_iteration_count, &relativePose_info.essential_matrix, relativePose_info.initial_residual_tolerance, false);
  relativePose_info.found_residual_precision = acRansacOut.first;

  if (relativePose_info.vec_inliers.size() < 2.5 * SolverType::MINIMUM_SAMPLES )
    return false; // no sufficient coverage (the model does not support enough samples)

  // estimation of the relative poses
  Mat3 R;
  Vec3 t;
  if (!estimate_Rt_fromE(
    K1, K2, x1, x2,
    relativePose_info.essential_matrix, relativePose_info.vec_inliers, &R, &t))
    return false; // cannot find a valid [R|t] couple that makes the inliers in front of the camera.

  // Store [R|C] for the second camera, since the first camera is [Id|0]
  relativePose_info.relativePose = geometry::Pose3(R, -R.transpose() * t);
  return true;
}

struct ResectionSquaredResidualError {
// Compute the residual of the projection distance(pt2D, Project(P,pt3D))
// Return the squared error
static double Error(const Mat34 & P, const Vec2 & pt2D, const Vec3 & pt3D){
  Vec2 x = Project(P, pt3D);
  return (x - pt2D).squaredNorm();
}
};

bool robustResection(
  const std::pair<size_t,size_t> & imageSize,
  const Mat & pt2D,
  const Mat & pt3D,
  std::vector<size_t> * pvec_inliers,
  const Mat3 * K,
  Mat34 * P,
  double * maxError,
  const size_t max_iteration,
  bool had_intrinsics)
{
  double dPrecision = std::numeric_limits<double>::infinity();//Square(16.0);
  size_t MINIMUM_SAMPLES = 0;
  // Classic resection (P matrix is unknown)
  if (!had_intrinsics)
  {
    typedef openMVG::resection::kernel::SixPointResectionSolver SolverType;
    MINIMUM_SAMPLES = SolverType::MINIMUM_SAMPLES;

    typedef ACKernelAdaptorResection<
      SolverType, ResectionSquaredResidualError, UnnormalizerResection, Mat34>
      KernelType;

    KernelType kernel(pt2D, imageSize.first, imageSize.second, pt3D);
    // Robustly estimation of the Projection matrix and it's precision
    std::pair<double,double> ACRansacOut = ACRANSAC(kernel, *pvec_inliers,
      max_iteration, P, dPrecision, true);
    *maxError = ACRansacOut.first;
  }
  else // K calibration matrix is known
  {
    typedef openMVG::euclidean_resection::P3PSolver SolverType;
    MINIMUM_SAMPLES = std::min(static_cast<size_t>(SolverType::MINIMUM_SAMPLES), size_t(6));

    typedef ACKernelAdaptorResection_K<
      SolverType, ResectionSquaredResidualError,
      UnnormalizerResection, Mat34>  KernelType;

    KernelType kernel(pt2D, pt3D, *K);
    // Robustly estimation of the Projection matrix and it's precision
    std::pair<double,double> ACRansacOut = ACRANSAC(kernel, *pvec_inliers,
      max_iteration, P, dPrecision, true);
    *maxError = ACRansacOut.first;
  }

  // Test if the mode support some points (more than those required for estimation)
  if (pvec_inliers->size() > 2.5 * MINIMUM_SAMPLES)
  {
    return true;
  }
  return false;
}



Vec2 disto_function(Vec2 ud, cv::Mat dist_coeff)
{

  double x_u = ud(0);
  double y_u = ud(1);

 double k1 = dist_coeff.at<double>(0,0);
 double k2 = dist_coeff.at<double>(0,1);
 double k3 = dist_coeff.at<double>(0,2);
 double t1 = dist_coeff.at<double>(0,3);
 double t2 = dist_coeff.at<double>(0,4);

         // Apply distortion (xd,yd) = disto(x_u,y_u)
  double r2 = x_u*x_u + y_u*y_u;
  double r4 = r2 * r2;
  double r6 = r4 * r2;
  double k_diff = (k1*r2 + k2*r4 + k3*r6);
  double t_x = t2 * (r2 + 2.0 * x_u*x_u) + 2.0 * t1 * x_u * y_u;
  double t_y = t1 * (r2 + 2.0 * y_u*y_u) + 2.0 * t2 * x_u * y_u;

  double x_d = x_u * k_diff + t_x;
  double y_d = y_u * k_diff + t_y;

  return Vec2(x_d,y_d);

}

cv::Point2f undistort_pt(cv::Point2f distorted, cv::Mat dist_coeff)
{
  const double epsilon = 1e-8; //criteria to stop the iteration

  Vec2 p(distorted.x,distorted.y);
  Vec2 p_u(distorted.x,distorted.y);

   while( ((p_u + disto_function(p_u, dist_coeff)) - p).lpNorm<1>() > epsilon ) //manhattan distance between the two points
   {
     p_u = p - disto_function( p_u, dist_coeff);
   }

   return cv::Point2f(p_u(0),p_u(1));
}


void undistort_points(vector<cv::Point2f> &distorted, cv::Mat K, cv::Mat dist_coeff)
{

     double k1 = dist_coeff.at<double>(0,0);
     double k2 = dist_coeff.at<double>(0,1);
     double k3 = dist_coeff.at<double>(0,2);
     double t1 = dist_coeff.at<double>(0,3);
     double t2 = dist_coeff.at<double>(0,4);

     //cout<<"Undistorting points with: "<<dist_coeff<<endl;

     vector<double> d_coeffs;

     d_coeffs.push_back(k1);
     d_coeffs.push_back(k2);
     d_coeffs.push_back(k3);
     d_coeffs.push_back(t1);
     d_coeffs.push_back(t2);

     vector<cv::Point2f> undistorted;


     //cout<<"Init undistorting.."<<endl;
     //cv::undistortPoints(distorted, undistorted, K, d_coeffs, cv::noArray() ,K); 

      for(int i=0;i<distorted.size();i++)
     {
        cv::Point2f d_off((distorted[i].x - K.at<double>(0,2))/K.at<double>(0,0), (distorted[i].y - K.at<double>(1,2))/K.at<double>(0,0));
        cv::Point2f u_d = undistort_pt(d_off,dist_coeff);
        u_d = u_d*K.at<double>(0,0);
        u_d = u_d + cv::Point2f(K.at<double>(0,2), K.at<double>(1,2));


        undistorted.push_back(u_d);
     }

     //   cout<<"Done undistorting.."<<endl;


        distorted = undistorted;
}


bool SolvePnP_AC_RANSAC(int img_width, int img_height, vector<cv::Point2f> points2D, vector<cv::Point3f> points3D, cv::Mat &K_cv, cv::Mat d_coeffs, cv::Matx34d &P_cv, double &threshold,bool had_intrinsics)
{

      std::pair<size_t,size_t> imageSize;
      // Create pt2D, and pt3D array
      Mat pt2D( 2, points2D.size());
      Mat pt3D( 3, points3D.size());
      std::vector<size_t> pvec_inliers;
      Mat3  K;
      Mat34  P;
      double maxError = std::numeric_limits<double>::max();
      size_t max_iteration = 512;//2048; //3048

      undistort_points(points2D, K_cv,d_coeffs);

/*Variable conversion from OpenCV to OpenMVG*/

    again:

    imageSize.first = img_width;
    imageSize.second = img_height;




    for(int i=0; i<3; i++)
        for(int j=0;j<3;j++)
        K(i,j) = K_cv.at<double>(i,j);

    for(int i=0; i<3; i++)
        for(int j=0;j<4;j++)
        P(i,j) = P_cv(i,j);

  for(int i=0;i<points2D.size();i++)
  {
     Vec2 feat2d;
     Vec3 feat3d;

     feat2d[0] = points2D[i].x; feat2d[1] = points2D[i].y;
     feat3d[0] = points3D[i].x; feat3d[1] = points3D[i].y; feat3d[2]= points3D[i].z;

    pt2D.col(i) = feat2d;
    pt3D.col(i) = feat3d;


  }

/*---------------------------------------------*/



      openMVG::sfm::robustResection(
      imageSize,
      pt2D,
      pt3D,
      &pvec_inliers,
      &K,
      &P,
      &maxError,
      max_iteration,
      had_intrinsics);




    for(int i=0; i<3; i++)
        for(int j=0;j<4;j++)
        P_cv(i,j) = P(i,j);

        if(had_intrinsics) //normalize it using K
        {
        cv::Mat P_norm = K_cv.inv()*cv::Mat(P_cv);


        for(int i=0;i<3;i++)
            for(int j=0;j<4;j++)
            P_cv(i,j) = P_norm.at<double>(i,j);

        }

    threshold = maxError;



    cv::Mat R,t,Kest;


    if(!had_intrinsics) //Decompose Projection matrix into K, R|t and update Kvec[id]
    {
        cout<<"Threshold without intrinsics: "<<threshold<<endl;
        cv::decomposeProjectionMatrix(P_cv,Kest,R,t,noArray(),noArray(),noArray(),noArray());

        P_cv(0,0) = R.at<double>(0,0); P_cv(0,1) = R.at<double>(0,1); P_cv(0,2) = R.at<double>(0,2);
        P_cv(1,0) = R.at<double>(1,0);  P_cv(1,1) = R.at<double>(1,1);  P_cv(1,2) = R.at<double>(1,2);
        P_cv(2,0) = R.at<double>(2,0);  P_cv(2,1) = R.at<double>(2,1);  P_cv(2,2) = R.at<double>(2,2);

        P_cv(0,3) = t.at<double>(0,0);  P_cv(1,3) = t.at<double>(0,1); P_cv(2,3) = t.at<double>(0,2);

        for(int i=0;i<Kest.rows;i++)
            for(int j=0;j<Kest.cols;j++)
                Kest.at<double>(i,j)/=Kest.at<double>(2,2); //scaling matrix


        cout<<" estimated K by decomposition = "<<Kest<<endl;// getchar();

        K_cv.at<double>(0,0) = abs(Kest.at<double>(0,0)); K_cv.at<double>(1,1) = abs(Kest.at<double>(0,0)); //update estimated focal


            had_intrinsics = true;


            goto again;


    }

      return true;

}

cv::Mat FindFundamentalMat_AC_RANSAC(vector<cv::Point2f> cam1, vector<cv::Point2f> cam2,cv::Mat K1_cv, cv::Mat K2_cv, int im1_w, int im1_h, int im2_w, int im2_h, double &threshold, vector<uchar> &vec_inliers)
{
  Mat3 K1,K2;
  cv::Mat fundamental_mat;
  fundamental_mat.create(3,3,CV_64F);
   Mat xI(2,cam1.size()), xJ(2,cam2.size());
  RelativePose_Info relativePose_info;
  std::pair<size_t, size_t>  size_ima1;
  std::pair<size_t, size_t> size_ima2;
  size_t max_iteration_count=600;

 size_ima1.first = im1_w;  size_ima1.second = im1_h;
 size_ima2.first = im2_w;  size_ima2.second = im2_h;

 /*K1(0,0)=1; K1(0,1)=0; K1(0,2)=0;
 K1(1,0)=0; K1(1,1)=1; K1(1,2)=0;
 K1(2,0)=0; K1(2,1)=0; K1(2,2)=1;

 K2(0,0)=1; K2(0,1)=0; K2(0,2)=0;
 K2(1,0)=0; K2(1,1)=1; K2(1,2)=0;
 K2(2,0)=0; K2(2,1)=0; K2(2,2)=1;
 */

 for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
    {
      K1(i,j) = K1_cv.at<double>(i,j);
      K2(i,j) = K2_cv.at<double>(i,j);
    }

 for(int i=0;i<cam1.size();i++)
 {
    Vec2 feat2d_1, feat2d_2;
    feat2d_1[0] = cam1[i].x;  feat2d_1[1] = cam1[i].y;
    feat2d_2[0] = cam2[i].x;  feat2d_2[1] = cam2[i].y;

    xI.col(i) = feat2d_1;
    xJ.col(i) = feat2d_2;
 }


 robustRelativePose(
   K1, K2,
   xI, xJ,
  relativePose_info,
  size_ima1,
  size_ima2,
  max_iteration_count);

threshold= relativePose_info.found_residual_precision;

for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
    fundamental_mat.at<double>(i,j) = relativePose_info.essential_matrix(i,j);


for(int i=0;i<vec_inliers.size() && i<relativePose_info.vec_inliers.size();i++)
    vec_inliers[i] = relativePose_info.vec_inliers[i];




return fundamental_mat;
}


} // namespace sfm
} // namespace openMVG



