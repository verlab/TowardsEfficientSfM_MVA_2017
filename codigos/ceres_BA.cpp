#include "ceres_BA.h"


//Part of this code was taken from OpenMVG lib

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "ceres/rotation.h"



/// Create the appropriate cost functor according the provided input camera intrinsic model
ceres::CostFunction * IntrinsicsToCostFunction(int opt_cam, double* observation)
{
  switch(opt_cam)
  {   
    case 0: //Pinhole_Simplified_K1
      return new ceres::AutoDiffCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Simplified, 2, 2, 6, 3>(
        new ResidualErrorFunctor_Pinhole_Intrinsic_Simplified(observation));    
    case 1: //Pinhole
      return new ceres::AutoDiffCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic, 2, 3, 6, 3>(
        new ResidualErrorFunctor_Pinhole_Intrinsic(observation));
    break;
    case 2: // Pinhole_K1
      return new ceres::AutoDiffCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1, 2, 4, 6, 3>(
        new ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1(observation));
    break;
    case 3: // Pinhole_K1..K3
      return new ceres::AutoDiffCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3, 2, 6, 6, 3>(
        new ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3(observation));
    break;
    case 4: // Pinhole_K1..K3_T1_T2
      return new ceres::AutoDiffCostFunction<ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2, 2, 9, 6, 3>(
        new ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2(observation));
    default:
      return NULL;
  }
}


Ceres_Bundler::BA_options::BA_options(bool bVerbose, int qtd_threads)
{

    _nbThreads = qtd_threads;

    _bVerbose = bVerbose;


  _bCeres_Summary = false;

  // Default configuration use a DENSE representation
  _linear_solver_type = ceres::DENSE_SCHUR;
  _preconditioner_type = ceres::JACOBI;
  // If Sparse linear solver are available
  // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
  if (true)//ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
  {
    _sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    _linear_solver_type = ceres::SPARSE_SCHUR;

    //cout<<"Using SuiteSparse (fastest)."<<endl; //getchar();
  }
  else
  {
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
      _sparse_linear_algebra_library_type = ceres::CX_SPARSE;
      _linear_solver_type = ceres::SPARSE_SCHUR;

       cout<<"USING SLOW LIB TO SPARSE LINEAR ALGEBRA"<<endl; //getchar();
    }
    else
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
      _sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
      _linear_solver_type = ceres::SPARSE_SCHUR;

       cout<<"USING SLOW LIB TO SPARSE LINEAR ALGEBRA"<<endl; //getchar();
    }
  }
}


Ceres_Bundler::Ceres_Bundler(
  Ceres_Bundler::BA_options options)
  //: _openMVG_options(options)
{

  _openMVG_options = options;
}

int Ceres_Bundler::isVisible(cloudPoint* cld)
{
  int nb_visib=0;

  for(int i=0;i<cld->owners.size();i++)
  {
    pKeypoint* pk = cld->owners[i];

    if(pk->img_owner->triangulated && pk->img_owner->pos != -1 && pk->active)
    {
        nb_visib++;
        if(nb_visib>1)
          break;
        

    }
  }

  return nb_visib;
}

cv::Mat Ceres_Bundler::Adjust(
  vector<imgMatches*> cams,     // the SfM scene to refine
  vector<Mat> &Kvec,
  vector<Mat> &Dvec,
  vector<int> idx_cams,
  vector<cloudPoint*> Cloud,
  bool bRefineRotations,   // tell if pose rotations will be refined
  bool bRefineTranslations,// tell if the pose translation will be refined
  bool bRefineCamera,
  bool bUniqueCamera, // True if the dataset has taken with a single camera and same calibration
  int __wsize)  // BA window size
{

  //----------
  // Add camera parameters
  // - intrinsics
  // - poses [R|t]

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  //----------
  cv::Mat uniqueCam;
  ceres::Problem problem;
  int wsize;

  if(__wsize>0) //secondary BA. only gets the size()-__wsize cameras (last ones) to perform BA.
    wsize = idx_cams.size()-__wsize;
    else
    wsize = 0;


  if(wsize<1 && __wsize != 0)
  {
    cout<<"Warning: LBA window size is too large for this scene."<<endl;

   uniqueCam = cv::Mat(1,2,CV_64F);
   uniqueCam.at<double>(0,0) = Kvec[idx_cams[0]].at<double>(0,0);
   uniqueCam.at<double>(0,1) = Dvec[idx_cams[0]].at<double>(0,0);
   return uniqueCam;
  }

  // Data wrapper for refinement:
 // Hash_Map<IndexT, std::vector<double> > map_intrinsics;
 // Hash_Map<IndexT, std::vector<double> > map_poses;

  vector<double*> intrinsics;
  vector<double*> poses;
  vector<double*> points3d;
  vector<double*> projs_2d;

  for(int i=0;i<cams.size();i++)
    cams[i]->pos=-1;

// Setup Poses data & subparametrization
  for(int i=wsize;i< idx_cams.size();i++)
  {
  	double R[9];
  	double angleAxis[3];
  	double t[3];
  	double* parameter_block = new double[6];


  	R[0]= cams[idx_cams[i]]->P_mat(0,0); R[3]= cams[idx_cams[i]]->P_mat(0,1); R[6]= cams[idx_cams[i]]->P_mat(0,2);
  	R[1]= cams[idx_cams[i]]->P_mat(1,0); R[4]= cams[idx_cams[i]]->P_mat(1,1); R[7]= cams[idx_cams[i]]->P_mat(1,2);
  	R[2]= cams[idx_cams[i]]->P_mat(2,0); R[5]= cams[idx_cams[i]]->P_mat(2,1); R[8]= cams[idx_cams[i]]->P_mat(2,2);

  	t[0] = cams[idx_cams[i]]->P_mat(0,3); t[1] = cams[idx_cams[i]]->P_mat(1,3); t[2] = cams[idx_cams[i]]->P_mat(2,3);


  	ceres::RotationMatrixToAngleAxis(R, angleAxis);

  	parameter_block[0] = angleAxis[0];
  	parameter_block[1] = angleAxis[1];
  	parameter_block[2] = angleAxis[2];
  	parameter_block[3] = t[0];
  	parameter_block[4] = t[1];
  	parameter_block[5] = t[2];

  	poses.push_back(parameter_block);
  	problem.AddParameterBlock(parameter_block, 6);


  	cams[idx_cams[i]]->pos=i-wsize; //

  	if (!bRefineTranslations && !bRefineRotations)
    {
      //set the whole parameter block as constant for best performance.
      problem.SetParameterBlockConstant(parameter_block);
    }
    else  {
      // Subset parametrization
      std::vector<int> vec_constant_extrinsic;
      if(!bRefineRotations)
      {
        vec_constant_extrinsic.push_back(0);
        vec_constant_extrinsic.push_back(1);
        vec_constant_extrinsic.push_back(2);
      }
      if(!bRefineTranslations)
      {
        vec_constant_extrinsic.push_back(3);
        vec_constant_extrinsic.push_back(4);
        vec_constant_extrinsic.push_back(5);
      }
      if (!vec_constant_extrinsic.empty())
      {
        ceres::SubsetParameterization *subset_parameterization =
          new ceres::SubsetParameterization(6, vec_constant_extrinsic);

        problem.SetParameterization(parameter_block, subset_parameterization);
      }
    }

  }


    unsigned int n;

    if(!bUniqueCamera)
    n = idx_cams.size();
    else
    n = wsize+1;

  // Setup Intrinsics data & subparametrization
    for(int i=wsize;i<n;i++)
    {

    	double * parameter_block = new double[2]; //f,k1
      std::vector<int> const_intrinsics;

    	parameter_block[0] = Kvec[idx_cams[i]].at<double>(0,0); //fx
    	parameter_block[1] = Dvec[idx_cams[i]].at<double>(0,0); //k1


    	intrinsics.push_back(parameter_block);
    	problem.AddParameterBlock(parameter_block, 2);

      if((bUniqueCamera && bRefineCamera) || (bUniqueCamera && wsize !=0 )) //if its 1 camera BA or LBA, fix intrinsics on unique-camera datasets
        problem.SetParameterBlockConstant(parameter_block);

       if(false)
      {
        const_intrinsics.push_back(1);
      }
      
      if (!const_intrinsics.empty())
      {
        ceres::SubsetParameterization *subset_parameterization =
          new ceres::SubsetParameterization(2, const_intrinsics);

        problem.SetParameterization(parameter_block, subset_parameterization);
      }

    }




  // Set a LossFunction to be less penalized by false measurements
  //  - set it to NULL if you don't want use a lossFunction.
  ceres::LossFunction * p_LossFunction = new ceres::HuberLoss(4.0); //(16.0);

  int num_of_residuals=0;
  // For all visibility add reprojections errors:
  for(int i=0;i<Cloud.size();i++)
  {

        if(bRefineRotations && bRefineTranslations && Cloud[i]->is_in_cloud &&  (isVisible(Cloud[i]) >1 || isVisible(Cloud[i])>0 && bRefineCamera))
        {

            double* X = new double[3];
            X[0] = Cloud[i]->x; X[1] = Cloud[i]->y; X[2] = Cloud[i]->z;
            points3d.push_back(X);
            problem.AddParameterBlock(X,3);/******/

            for(int j=0;j<Cloud[i]->owners.size();j++)
            {
                double* pt = new double[2];

                projs_2d.push_back(pt);

                pKeypoint* pk = Cloud[i]->owners[j];

                pt[0] = pk->pt.x - Kvec[pk->img_owner->id].at<double>(0,2); // translade to cx, cy (Pinhole Simplified)
                pt[1] = pk->pt.y - Kvec[pk->img_owner->id].at<double>(1,2);

              if(pk->img_owner->triangulated && pk->img_owner->pos != -1 && pk->active) //if this residual is in the current scene
              {

                ceres::CostFunction* cost_function =
                IntrinsicsToCostFunction(0, pt); //f k1

                if(!bUniqueCamera)
                {
                  problem.AddResidualBlock(cost_function,
                  p_LossFunction,
                  intrinsics[pk->img_owner->pos],
                  poses[pk->img_owner->pos],
                  X); //Do we need to copy 3D point to avoid false motion, if failure ?
                }
                else
                {
                  problem.AddResidualBlock(cost_function,
                  p_LossFunction,
                  intrinsics[0], //UNIQUE INTRINSICS FOR ALL CAMERAS
                  poses[pk->img_owner->pos],
                  X); //Do we need to copy 3D point to avoid false motion, if failure ?                 
                }


                num_of_residuals++;

              }


            }

             if (bRefineCamera) // there is only one camera to adjust -- fix all 3D points
                problem.SetParameterBlockConstant(X);

              else if(__wsize>0) // it is local BA -- Fix points that has flag = 1
              {
                if(Cloud[i]->can_ba == 1)//if this 3D point was seen by a camera
                    problem.SetParameterBlockConstant(X);
                else 
                    Cloud[i]->can_ba = 1; //fix 3D point if it was already bundle-adjusted
                
              }
              else // global BA -> set flag = 1 to seen 3D pts
              {
                  Cloud[i]->can_ba=1;              
              }
      }

  }
  //cout<<"The number of residuals is "<<num_of_residuals<<endl; //getchar();

  // Configure a BA engine and run it
  //  Make Ceres automatically detect the bundle structure.
  ceres::Solver::Options options;

  if(true) //SPARSE_CHOLESKY
  {
      options.preconditioner_type = ceres::SCHUR_JACOBI;//_openMVG_options._preconditioner_type;
      options.linear_solver_type = _openMVG_options._linear_solver_type;
      options.sparse_linear_algebra_library_type = _openMVG_options._sparse_linear_algebra_library_type;
      //options.min_relative_decrease = 0.03;
      options.max_linear_solver_iterations = 100;

      if(bRefineCamera)
       options.max_num_iterations = 100;
      else
        options.max_num_iterations = 25;

      if(!bRefineCamera)
      {
        options.parameter_tolerance = 1e-8;
        options.function_tolerance = 1e-4; // 1e1;
      }

  }
  else if(false)//CONJUGATE GRADIENTS
  {

  options.preconditioner_type =  ceres::JACOBI;
  options.linear_solver_type = ceres::CGNR;
  options.sparse_linear_algebra_library_type = _openMVG_options._sparse_linear_algebra_library_type;

  }
  else //bundler options
  {
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.sparse_linear_algebra_library_type = _openMVG_options._sparse_linear_algebra_library_type;
    options.parameter_tolerance = 1e-8;
    options.function_tolerance = 1e-4; // 1e1;
  }

  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;
  options.num_threads = _openMVG_options._nbThreads;
  options.num_linear_solver_threads = _openMVG_options._nbThreads;

  // Solve BA
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (_openMVG_options._bCeres_Summary)
    std::cout << summary.FullReport() << std::endl;

  // If no error, get back refined parameters
  if (!summary.IsSolutionUsable())
  {
    //if (_openMVG_options._bVerbose)
      std::cout << "Bundle Adjustment failed." << std::endl; getchar();
      return cv::Mat();
  }
  else // Solution is usable
  {
    if (_openMVG_options._bVerbose && bRefineRotations && bRefineTranslations)
    {
      // Display statistics about the minimization
      std::cout << std::endl
        << "Bundle Adjustment statistics (approximated RMSE):\n"
        //<< " #views: " << sfm_data.views.size() << "\n"
        //<< " #poses: " << sfm_data.poses.size() << "\n"
        //<< " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
        //<< " #tracks: " << sfm_data.structure.size() << "\n"
        << " #residuals: " << summary.num_residuals << "\n"
        << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "\n"
        << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n"
        << " Time (s): " << summary.total_time_in_seconds << "\n"
        << " Num Iters: " << summary.iterations.size() << "\n"
        //<<summary.FullReport()<<endl
        << std::endl;

       // getchar();
    }
    else if(!bRefineRotations && !bRefineTranslations)
    {
      cout << "  #projections: " << num_of_residuals << " "
        << " Initial RMSE: " << std::sqrt( summary.initial_cost / summary.num_residuals) << "-->"
        << " Final RMSE: " << std::sqrt( summary.final_cost / summary.num_residuals) << "\n";
    }

    // Update camera poses with refined data
    /*if (bRefineRotations || bRefineTranslations)
    {
      for (Poses::iterator itPose = sfm_data.poses.begin();
        itPose != sfm_data.poses.end(); ++itPose)
      {
        const IndexT indexPose = itPose->first;

        Mat3 R_refined;
        ceres::AngleAxisToRotationMatrix(&map_poses[indexPose][0], R_refined.data());
        Vec3 t_refined(map_poses[indexPose][3], map_poses[indexPose][4], map_poses[indexPose][5]);
        // Update the pose
        Pose3 & pose = itPose->second;
        pose = Pose3(R_refined, -R_refined.transpose() * t_refined);
      }
    }

    // Update camera intrinsics with refined data
    if (bRefineIntrinsics)
    {
      for (Intrinsics::iterator itIntrinsic = sfm_data.intrinsics.begin();
        itIntrinsic != sfm_data.intrinsics.end(); ++itIntrinsic)
      {
        const IndexT indexCam = itIntrinsic->first;

        const std::vector<double> & vec_params = map_intrinsics[indexCam];
        itIntrinsic->second.get()->updateFromParams(vec_params);
      }
    }
    */


    if (bRefineRotations || bRefineTranslations)
    {
     for(int i=wsize;i< idx_cams.size();i++) //Recover Poses
     {

        double R[9];
        double angleAxis[3];

        angleAxis[0] = poses[i-wsize][0]; angleAxis[1] = poses[i-wsize][1]; angleAxis[2] = poses[i-wsize][2];

        ceres::AngleAxisToRotationMatrix(angleAxis, R);

        cams[idx_cams[i]]->P_mat(0,0) = R[0]; cams[idx_cams[i]]->P_mat(0,1) = R[3]; cams[idx_cams[i]]->P_mat(0,2) = R[6];
        cams[idx_cams[i]]->P_mat(1,0) = R[1]; cams[idx_cams[i]]->P_mat(1,1) = R[4]; cams[idx_cams[i]]->P_mat(1,2) = R[7];
        cams[idx_cams[i]]->P_mat(2,0) = R[2]; cams[idx_cams[i]]->P_mat(2,1) = R[5]; cams[idx_cams[i]]->P_mat(2,2) = R[8];

         cams[idx_cams[i]]->P_mat(0,3) = poses[i-wsize][3];  cams[idx_cams[i]]->P_mat(1,3) = poses[i-wsize][4];  cams[idx_cams[i]]->P_mat(2,3) = poses[i-wsize][5];

        delete poses[i-wsize];
      }
    }


      for (int i=wsize; i<idx_cams.size();i++) // RecovEr INTRINSICS
      {

        if(!bUniqueCamera)
        {

        Kvec[idx_cams[i]].at<double>(0,0) = intrinsics[i-wsize][0]; //f
        Kvec[idx_cams[i]].at<double>(1,1) = intrinsics[i-wsize][0];  //f

        Dvec[idx_cams[i]].at<double>(0,0) = intrinsics[i-wsize][1]; //K1

        delete intrinsics[i-wsize];
        }
        else
        {          

        Kvec[idx_cams[i]].at<double>(0,0) = intrinsics[0][0]; //f
        Kvec[idx_cams[i]].at<double>(1,1) = intrinsics[0][0]; //f

        Dvec[idx_cams[i]].at<double>(0,0) = intrinsics[0][1]; //K1


        }
      }

      if(bUniqueCamera)
      {
        uniqueCam = cv::Mat(1,2,CV_64F);
         for(int j=0;j<2;j++)
           uniqueCam.at<double>(0,j) = intrinsics[0][j];

         for(int j=0;j<intrinsics.size();j++)
            delete intrinsics[j];
        
      }



      int j=0;

      if(bRefineRotations && bRefineTranslations)
      {
        for(int i=0;i<Cloud.size();i++) //Recover 3D points
          if(Cloud[i]->is_in_cloud && (isVisible(Cloud[i]) >1 || isVisible(Cloud[i])>0 && bRefineCamera) )
          {
            Cloud[i]->x =points3d[j][0];
            Cloud[i]->y =points3d[j][1];
            Cloud[i]->z =points3d[j][2];

            delete points3d[j]; //deallocate 3D positions

            j++;
          }
      }


    for(long i=0;i<projs_2d.size();i++) //deallocate 2d projs
    	delete projs_2d[i];




  }

 return uniqueCam;
}




