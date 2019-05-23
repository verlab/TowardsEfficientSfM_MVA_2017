
//Part of this code was taken of OpenMVG lib which is
// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef VERLAB_SFM_CERES_BA_HPP
#define VERLAB_SFM_CERES_BA_HPP

#include "ceres_camera_functor.hpp"
#include "ceres/ceres.h"
#include "Utilitarios.h"



/// Create the appropriate cost functor according the provided input camera intrinsic model
ceres::CostFunction * IntrinsicsToCostFunction(
  int opt_cam,
  double* observation);

class Ceres_Bundler
{
  public:
  struct BA_options
  {
    bool _bVerbose;
    unsigned int _nbThreads;
    bool _bCeres_Summary;
    ceres::LinearSolverType _linear_solver_type;
    ceres::PreconditionerType _preconditioner_type;
    ceres::SparseLinearAlgebraLibraryType _sparse_linear_algebra_library_type;

    BA_options(const bool bVerbose = true, int qtd_threads = 20);
  };
 // private:


  public:
  	BA_options _openMVG_options;

  Ceres_Bundler(Ceres_Bundler::BA_options options = BA_options());


  cv::Mat Adjust(
   vector<imgMatches*> sfm_data,            // the SfM scene to refine
   vector<Mat> &Kvec,
   vector<Mat> &Dvec,
   vector<int> idx_cams,
   vector<cloudPoint*> Cloud,
    bool bRefineRotations = true,   // tell if pose rotations will be refined
    bool bRefineTranslations = true,// tell if the pose translation will be refined
    bool bRefineCamera = true, // tell if it's only one camera to optimize
    bool bUniqueCamera = true, // tell if we are using local ba approach -- if it is false, optimize all the structure
    int wsize = 0);  // tell the size of the secondary window -- if it is 0, then its a primary ba -- in case we are using a local ba approach --

   int isVisible(cloudPoint* cld);

   bool Local_Adjustment(
   vector<imgMatches*> sfm_data,            // the SfM scene to refine
   vector<Mat> &Kvec,
   vector<Mat> &Dvec,
   vector<int> idx_cams,
   vector<cloudPoint*> Cloud,
   bool bUniqueCamera,
    int wsize = 0);  // tell the size of the secondary window -- if it is 0, then its a primary ba -- in case we are using a local ba approach --
};



#endif // VERLAB_SFM_CERES_BA_HPP
