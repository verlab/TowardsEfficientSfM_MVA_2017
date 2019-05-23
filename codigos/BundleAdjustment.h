// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//#define V3DLIB_ENABLE_SUITESPARSE


//#include "Utilitarios.h" //
//#include "Math/v3d_linear.h"
//#include "Base/v3d_vrmlio.h"
//#include "Geometry/v3d_metricbundle.h"


#include <cvsba/cvsba.h>




//#include "BundleAdjustment.h"

//using namespace V3D;
using namespace std;
using namespace cv;


/*void showErrorStatistics(double const f0,
						StdDistortionFunction const& distortion,
						vector<CameraMatrix> const& cams,
						vector<Vector3d> const& Xs,
						vector<Vector2d> const& measurements,
						vector<int> const& correspondingView,
						vector<int> const& correspondingPoint)
	{
		int const K = measurements.size();

		double meanReprojectionError = 0.0;
		for (int k = 0; k < K; ++k)
		{
			int const i = correspondingView[k];
			int const j = correspondingPoint[k];
			Vector2d p = cams[i].projectPoint(distortion, Xs[j]);

			double reprojectionError = norm_L2(f0 * (p - measurements[k]));
			meanReprojectionError += reprojectionError;

		}
		cout << "mean reprojection error (in pixels): " << meanReprojectionError/K << endl;
	}
 // end namespace <>
*/
//count number of 2D measurements
long Count2DMeasurements(vector<cloudPoint*> pointcloud)
{
	long K = 0;

	for(int i=0;i<pointcloud.size();i++)
	{
	    if(pointcloud[i]->is_in_cloud)
	       K+=pointcloud[i]->owners.size();
	}
	return K;
}

long Count3DMeasurements(vector<cloudPoint*> pointcloud)
{
    long K =0;

    	for(int i=0;i<pointcloud.size();i++)
	{
	    if(pointcloud[i]->is_in_cloud)
	       K++;
	}

	return K;
}

void adjust_ba(vector<cloudPoint*> pointcloud, Mat& cam_matrix, vector<imgMatches*> all_Pmats)
{

  /*  vector<imgMatches*> Pmats;

    for(int i=0;i<all_Pmats.size();i++)
      if(all_Pmats[i]->triangulated)
      Pmats.push_back(all_Pmats[i]);


	long N = Pmats.size(), M = Count3DMeasurements(pointcloud), K=0 ;

	cout << "(cams) = " << N << " (3d points) = " << M << " (2d measurements) = " << K << endl;

	StdDistortionFunction distortion;

	//conver camera intrinsics to BA datastructs
	Matrix3x3d KMat;
	makeIdentityMatrix(KMat);
	KMat[0][0] = cam_matrix.at<double>(0,0); //fx
	KMat[1][1] = cam_matrix.at<double>(1,1); //fy
	KMat[0][1] = cam_matrix.at<double>(0,1); //skew
	KMat[0][2] = cam_matrix.at<double>(0,2); //ppx
	KMat[1][2] = cam_matrix.at<double>(1,2); //ppy

	double const f0 = KMat[0][0];
	cout << "intrinsic before bundle = "; displayMatrix(KMat);
	Matrix3x3d Knorm = KMat;
	// Normalize the intrinsic to have unit focal length.
	scaleMatrixIP(1.0/f0, Knorm);
	Knorm[2][2] = 1.0;



	//conver 3D point cloud to BA datastructs
	vector<Vector3d > Xs(M);
	for (long i = 0, j=0; i < pointcloud.size();i++)
	{
      if(pointcloud[i]->is_in_cloud)
       {
        Xs[j][0] = pointcloud[i]->x;
		Xs[j][1] = pointcloud[i]->y;
		Xs[j][2] = pointcloud[i]->z;
		pointcloud[i]->idx = j;  j++;
       }
	}
	cout << "Read the 3D points." << endl;


	vector<Vector2d > measurements;
	vector<int> correspondingView;
	vector<int> correspondingPoint;

	//measurements.reserve(K);
	//correspondingView.reserve(K);
	//correspondingPoint.reserve(K);


	//convert params to BA datastructs
	vector<CameraMatrix> cams(N);
	for (int i = 0; i < N; ++i)
	{

		Matrix3x3d R;
		Vector3d T;

		Matx34d P = Pmats[i]->P_mat;

		R[0][0] = P(0,0); R[0][1] = P(0,1); R[0][2] = P(0,2); T[0] = P(0,3);
		R[1][0] = P(1,0); R[1][1] = P(1,1); R[1][2] = P(1,2); T[1] = P(1,3);
		R[2][0] = P(2,0); R[2][1] = P(2,1); R[2][2] = P(2,2); T[2] = P(2,3);


		cams[i].setIntrinsic(Knorm);
		cams[i].setRotation(R);
		cams[i].setTranslation(T);


         for(int j=0;j<Pmats[i]->pkeypoints.size();j++)
         {
            if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            {


              Vector3d p, np;

                        Point cvp = Pmats[i]->pkeypoints[j]->pt; //adding points of this view
                        p[0] = cvp.x;
                        p[1] = cvp.y;
                        p[2] = 1.0;


                            // Normalize the measurements to match the unit focal length.
                            scaleVectorIP(1.0/f0, p);
                            measurements.push_back(Vector2d(p[0], p[1]));
                            correspondingView.push_back(i);
                            correspondingPoint.push_back(Pmats[i]->pkeypoints[j]->cloud_p->idx);

            }

         }
	}
	cout << "Cameras read." << endl;



	K = measurements.size();

	cout << "Read " << K << " valid 2D measurements." << endl;

	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

//	V3D::optimizerVerbosenessLevel = 1;
	double const inlierThreshold = 2.0 / fabs(f0);

	Matrix3x3d K0 = cams[0].getIntrinsic();
	cout << "K0 = "; displayMatrix(K0);

	bool good_adjustment = false;
	{
		ScopedBundleExtrinsicNormalizer extNorm(cams, Xs);
		ScopedBundleIntrinsicNormalizer intNorm(cams,measurements,correspondingView);
		CommonInternalsMetricBundleOptimizer opt(V3D::FULL_BUNDLE_FOCAL_LENGTH_PP, inlierThreshold, K0, distortion, cams, Xs,
												 measurements, correspondingView, correspondingPoint);


		opt.tau = 1e-3;
		opt.maxIterations = 50;
		opt.minimize();

		cout << "optimizer status = " << opt.status << endl;

		good_adjustment = (opt.status != 2);
	}

	cout << "refined K = "; displayMatrix(K0);

	for (int i = 0; i < N; ++i) cams[i].setIntrinsic(K0);

	Matrix3x3d Knew = K0;
	scaleMatrixIP(f0, Knew);
	Knew[2][2] = 1.0;
	cout << "Knew = "; displayMatrix(Knew);

	showErrorStatistics(f0, distortion, cams, Xs, measurements, correspondingView, correspondingPoint);

	if(good_adjustment)
	{

		//extract 3D points
		for (unsigned int i = 0; i < pointcloud.size(); i++)
		{
		    if(pointcloud[i]->is_in_cloud)
		    {

            Vector3d pt; pt[0]= Xs[pointcloud[i]->idx][0]; pt[1]=Xs[pointcloud[i]->idx][1]; pt[2]=Xs[pointcloud[i]->idx][2];
            //scaleVectorIP(f0, pt);

            pointcloud[i]->x = pt[0];//Xs[pointcloud[i]->idx][0];
			pointcloud[i]->y = pt[1];//Xs[pointcloud[i]->idx][1];
			pointcloud[i]->z = pt[2];//Xs[pointcloud[i]->idx][2];
		    }
		}

		//extract adjusted cameras
		for (int i = 0; i < N; i++)
		{
			Matrix3x3d R = cams[i].getRotation();
			Vector3d T = cams[i].getTranslation();

			Matx34d P;
			P(0,0) = R[0][0]; P(0,1) = R[0][1]; P(0,2) = R[0][2]; P(0,3) = T[0];
			P(1,0) = R[1][0]; P(1,1) = R[1][1]; P(1,2) = R[1][2]; P(1,3) = T[1];
			P(2,0) = R[2][0]; P(2,1) = R[2][1]; P(2,2) = R[2][2]; P(2,3) = T[2];

			Pmats[i]->P_mat = P;
		}


		//extract camera intrinsics
		cam_matrix.at<double>(0,0) = Knew[0][0];
		cam_matrix.at<double>(0,1) = Knew[0][1];
		cam_matrix.at<double>(0,2) = Knew[0][2];
		cam_matrix.at<double>(1,1) = Knew[1][1];
		cam_matrix.at<double>(1,2) = Knew[1][2];
	}
*/
}

void adjust_ba_opencv(vector<cloudPoint*> pointcloud, Mat& cam_matrix, vector<imgMatches*> all_Pmats, int rows, int cols)
{

  /*  vector<imgMatches*> Pmats;

    for(int i=0;i<all_Pmats.size();i++)
      if(all_Pmats[i]->triangulated)
       Pmats.push_back(all_Pmats[i]);


	long N = Pmats.size(), M = Count3DMeasurements(pointcloud), K=0 ;


    vector<Point3d> points;		// positions of points in global coordinate system (input and output)
	vector<vector<Point2d> > imagePoints; // projections of 3d points for every camera
	vector<vector<int> > visibility; // visibility of 3d points for every camera
	vector<Mat> cameraMatrix;	// intrinsic matrices of all cameras (input and output)
	vector<Mat> R;				// rotation matrices of all cameras (input and output)
	vector<Mat> T;				// translation vector of all cameras (input and output)
	vector<Mat> distCoeffs;		// distortion coefficients of all cameras (input and output)
    cv::TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 70, 0.01);




	//loading 3D points
	for (long i = 0, j=0; i < pointcloud.size();i++)
	{
      if(pointcloud[i]->is_in_cloud)
       {
        Point3d pt3d;

        pt3d.x = pointcloud[i]->x;
		pt3d.y = pointcloud[i]->y;
		pt3d.z = pointcloud[i]->z;
		points.push_back(pt3d);
        pointcloud[i]->idx = j;  j++;


       }
	}

	/*for (int i=0; i<N; i++)
	{
		cameraMatrix[i] = cam_matrix;

		Matx34d& P = Pmats[i]->P_mat;

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R[i] = camR;
		T[i] = camT;

//		distCoeffs[i] = Mat(); //::zeros(4,1, CV_64FC1);
	}*/

	/*for (int i = 0; i < N; i++) //for all cameras
	{

		cameraMatrix.push_back(cam_matrix);
		distCoeffs.push_back((cv::Mat_<double>(5,1) << 0, 0, 0, 0, 0));

		Matx34d P = Pmats[i]->P_mat;
		Mat_<double> KP1 = cam_matrix * Mat(P); //Projection matrix

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R.push_back(camR);
		T.push_back(camT);

         vector<Point2d> img_pts(points.size());
         vector<int> visib_cam(points.size(),0);

         for(long j=0,k=0;j<Pmats[i]->pkeypoints.size();j++)
         {


            if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            {
                 cloudPoint * cp = Pmats[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 //Point2d cvp(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                img_pts[Pmats[i]->pkeypoints[j]->cloud_p->idx]=Pmats[i]->pkeypoints[j]->pt;

               // if(cvp.x >=0 && cvp.y>=0 && cvp.x<cols && cvp.y< rows)
                visib_cam[Pmats[i]->pkeypoints[j]->cloud_p->idx] = 1;


            }

         }

         imagePoints.push_back(img_pts);
         visibility.push_back(visib_cam);
	}

	cv::LevMarqSparse::bundleAdjust(points,imagePoints,visibility,cameraMatrix,R,T,distCoeffs,criteria);

	for (long i = 0; i < pointcloud.size(); i++)
	{
	     if(pointcloud[i]->is_in_cloud)
       {
	    Point3d ba_point = points[pointcloud[i]->idx];

		pointcloud[i]->x = ba_point.x;
		pointcloud[i]->y = ba_point.y;
		pointcloud[i]->z = ba_point.z;

       }
	}

    for (int i = 0; i < N; i++)
        {
            Matx34d P;
            P(0,0) = R[i].at<double>(0,0); P(0,1) = R[i].at<double>(0,1); P(0,2) = R[i].at<double>(0,2); P(0,3) = T[i].at<double>(0);
            P(1,0) = R[i].at<double>(1,0); P(1,1) = R[i].at<double>(1,1); P(1,2) = R[i].at<double>(1,2); P(1,3) = T[i].at<double>(1);
            P(2,0) = R[i].at<double>(2,0); P(2,1) = R[i].at<double>(2,1); P(2,2) = R[i].at<double>(2,2); P(2,3) = T[i].at<double>(2);

            Pmats[i]->P_mat = P;
        }

  cam_matrix = cameraMatrix[0];

  */

}

void adjust_ba_cvsba(vector<cloudPoint*> pointcloud, Mat& cam_matrix, vector<imgMatches*> all_Pmats, int rows, int cols, bool local_opt, vector<int> idx_cams)
{

    vector<imgMatches*> Pmats;
    //cv::Mat Kinv = cam_matrix.inv();

    if(!local_opt)
    {

    for(int i=0;i<all_Pmats.size();i++)
      if(all_Pmats[i]->triangulated)
       Pmats.push_back(all_Pmats[i]);

    }
    else
    {
        for(int i=0;i<idx_cams.size();i++)
        {
             if(all_Pmats[idx_cams[i]]->triangulated)
                 Pmats.push_back(all_Pmats[idx_cams[i]]);
        }

    }


	long N = Pmats.size(), M = Count3DMeasurements(pointcloud), K=0 ;


    vector<Point3d> points;		// positions of points in global coordinate system (input and output)
	vector<vector<Point2d> > imagePoints; // projections of 3d points for every camera
	vector<vector<int> > visibility; // visibility of 3d points for every camera
	vector<Mat> cameraMatrix;	// intrinsic matrices of all cameras (input and output)
	vector<Mat> R;				// rotation matrices of all cameras (input and output)
	vector<Mat> T;				// translation vector of all cameras (input and output)
	vector<Mat> distCoeffs;		// distortion coefficients of all cameras (input and output)
    cv::TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 70, 1e-10);


//////////////local optima



long cloud_idx=0;

if(local_opt)
{

for (int i=0;i<Pmats.size();i++) //
  for(long j=0;j<Pmats[i]->pkeypoints.size();j++)
    {
        if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud && Pmats[i]->pkeypoints[j]->cloud_p->idx==-1)
            {
            Point3d pt3d;

            pt3d.x = Pmats[i]->pkeypoints[j]->cloud_p->x;
            pt3d.y = Pmats[i]->pkeypoints[j]->cloud_p->y;
            pt3d.z = Pmats[i]->pkeypoints[j]->cloud_p->z;
            points.push_back(pt3d);
            Pmats[i]->pkeypoints[j]->cloud_p->idx = cloud_idx;  cloud_idx++;
            }

    }
    M = cloud_idx;
}
////////////////////right//////////////
else
{


	//loading 3D points
	for (long i = 0, j=0; i < pointcloud.size();i++)
	{
      if(pointcloud[i]->is_in_cloud)
       {
        Point3d pt3d;

        pt3d.x = pointcloud[i]->x;
		pt3d.y = pointcloud[i]->y;
		pt3d.z = pointcloud[i]->z;
		points.push_back(pt3d);
        pointcloud[i]->idx = j;  j++;


       }
	}
}

	/*for (int i=0; i<N; i++)
	{
		cameraMatrix[i] = cam_matrix;

		Matx34d& P = Pmats[i]->P_mat;

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R[i] = camR;
		T[i] = camT;

//		distCoeffs[i] = Mat(); //::zeros(4,1, CV_64FC1);
	}*/
      // cout<<"foi"<<endl;
	for (int i = 0; i < N; i++) //for all cameras
	{

		cameraMatrix.push_back(cam_matrix);
		distCoeffs.push_back((cv::Mat_<double>(5,1) << 0, 0, 0, 0, 0));


		Matx34d P = Pmats[i]->P_mat;
		Mat_<double> KP1 = cam_matrix * Mat(P); //Projection matrix

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R.push_back(camR);
		T.push_back(camT);

         vector<Point2d> img_pts(points.size());
         vector<int> visib_cam(points.size(),0);

         for(int j=0,k=0;j<Pmats[i]->pkeypoints.size();j++)
         {


            if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            {
                 cloudPoint * cp = Pmats[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                Point2d cvp (Pmats[i]->pkeypoints[j]->pt.x,Pmats[i]->pkeypoints[j]->pt.y); //adding points of this view
                img_pts[Pmats[i]->pkeypoints[j]->cloud_p->idx]=cvp;

                 //if(cvp.x >=0 && cvp.y>=0 && cvp.x<cols && cvp.y< rows)
                visib_cam[Pmats[i]->pkeypoints[j]->cloud_p->idx] = 1;

                 //cout<<"Predicted:" <<cvp0 << "Actual:" << cvp <<endl;
                 Pmats[i]->pkeypoints[j]->cloud_p->reproj_error+=norm(cvp0-cvp);

            }

         }

         imagePoints.push_back(img_pts);
         visibility.push_back(visib_cam);
	}


    cvsba::Sba sba;
    cvsba::Sba::Params param;

    param.fixedDistortion=5;
    //param.fixedIntrinsics=5;
    param.iterations=35;
    param.verbose = false;
    param.minError = 1e-10;// 0.002; //1e-10
    param.type = sba.MOTIONSTRUCTURE;

    sba.setParams(param);


    try
    {

     //ncon(3dpoints), mcon(cameras)
     sba.run(0,0,points, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);

     std::cout<<"Initial error="<<sba.getInitialReprjError()<<". "<<
                "Final error="<<sba.getFinalReprjError()<<std::endl;


if(!local_opt)
{


	for (long i = 0; i <  pointcloud.size(); i++) //global
	{
	     if(pointcloud[i]->is_in_cloud)
       {
	    Point3d ba_point = points[pointcloud[i]->idx];

		pointcloud[i]->x = ba_point.x;
		pointcloud[i]->y = ba_point.y;
		pointcloud[i]->z = ba_point.z;

       }
	}


}
else
{


//local


for (int i=0;i<Pmats.size();i++) //
  for(long j=0;j<Pmats[i]->pkeypoints.size();j++)
    {
        if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud && Pmats[i]->pkeypoints[j]->cloud_p->idx!=-1)
            {
              Point3d ba_point = points[Pmats[i]->pkeypoints[j]->cloud_p->idx];
                Pmats[i]->pkeypoints[j]->cloud_p->x = ba_point.x;
                Pmats[i]->pkeypoints[j]->cloud_p->y = ba_point.y;
                Pmats[i]->pkeypoints[j]->cloud_p->z = ba_point.z;
                Pmats[i]->pkeypoints[j]->cloud_p->idx = -1;
            }
    }

}
///////

    for (int i = 0; i < N; i++)
        {
            Matx34d P;
            P(0,0) = R[i].at<double>(0,0); P(0,1) = R[i].at<double>(0,1); P(0,2) = R[i].at<double>(0,2); P(0,3) = T[i].at<double>(0);
            P(1,0) = R[i].at<double>(1,0); P(1,1) = R[i].at<double>(1,1); P(1,2) = R[i].at<double>(1,2); P(1,3) = T[i].at<double>(1);
            P(2,0) = R[i].at<double>(2,0); P(2,1) = R[i].at<double>(2,1); P(2,2) = R[i].at<double>(2,2); P(2,3) = T[i].at<double>(2);

            Pmats[i]->P_mat = P;
        }

  cam_matrix = cameraMatrix[0];
    }

    catch(...)
    {
        cout<<"Something went wrong with BA. Maybe a wrong camera was estimated"<<endl;
    }
}



void adjust_camera_cvsba(Mat& cam_matrix, Mat& dist_coeff, imgMatches* camera,double max_error)
{

    vector<cloudPoint*> cloud_aux;
    int i=0;
    //cv::Mat Kinv = cam_matrix.inv();



    vector<Point3d> points;		// positions of points in global coordinate system (input and output)
	vector<vector<Point2d> > imagePoints; // projections of 3d points for every camera
	vector<vector<int> > visibility; // visibility of 3d points for every camera
	vector<Mat> cameraMatrix;	// intrinsic matrices of all cameras (input and output)
	vector<Mat> R;				// rotation matrices of all cameras (input and output)
	vector<Mat> T;				// translation vector of all cameras (input and output)
	vector<Mat> distCoeffs;		// distortion coefficients of all cameras (input and output)
    cv::TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 70, 1e-10);




    long cloud_idx=0;


	//loading 3D points
        for(int j=0;j<camera->pkeypoints.size();j++)
         {


            if(camera->pkeypoints[j]->cloud_p!=NULL && camera->pkeypoints[j]->cloud_p->is_in_cloud)
            {
             cloudPoint * cp = camera->pkeypoints[j]->cloud_p;



                    points.push_back(Point3d(cp->x,cp->y,cp->z));
                    cloud_aux.push_back(cp);

                    cp->idx = cloud_idx++;



            }
        }




		cameraMatrix.push_back(cam_matrix);
		distCoeffs.push_back(dist_coeff);


		Matx34d P = camera->P_mat;
		Mat_<double> KP1 = cam_matrix * Mat(P); //Projection matrix

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R.push_back(camR);
		T.push_back(camT);

         vector<Point2d> img_pts(points.size());
         vector<int> visib_cam(points.size(),0);



         for(int j=0,k=0;j<camera->pkeypoints.size();j++)
         {


            if(camera->pkeypoints[j]->cloud_p!=NULL && camera->pkeypoints[j]->cloud_p->is_in_cloud)
            {
                 cloudPoint * cp = camera->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 Mat_<double> XK(3,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;


                 				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
                 Point2d cvp (camera->pkeypoints[j]->pt.x,camera->pkeypoints[j]->pt.y); //adding points of this view



                    img_pts[camera->pkeypoints[j]->cloud_p->idx]=cvp;
                    visib_cam[camera->pkeypoints[j]->cloud_p->idx] = 1;



                 //camera->pkeypoints[j]->cloud_p->reproj_error+=norm(cvp0-cvp);

            }

         }

         imagePoints.push_back(img_pts);
         visibility.push_back(visib_cam);



    cvsba::Sba sba;
    cvsba::Sba::Params param;

    param.fixedDistortion=5;//3 //optimize k1 and k2
    param.fixedIntrinsics=5;//4 //optimize fx
    param.iterations=60;
    param.verbose = false;
    param.minError = 1e-10;// 0.002; //1e-10
    param.type = sba.MOTION;

    sba.setParams(param);


    try
    {

     //ncon(3dpoints), mcon(cameras)
     sba.run(points.size(),0,points, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);

     std::cout<<"Adjusting camera motion: Initial error="<<sba.getInitialReprjError()<<". "<<
                "Final error="<<sba.getFinalReprjError()<<std::endl;




	for (long i = 0; i <  cloud_aux.size(); i++) //global
	{

	    Point3d ba_point = points[cloud_aux[i]->idx];
        cloud_aux[i]->idx = -1;
		//cloud_aux[i]->x = ba_point.x;
		//cloud_aux[i]->y = ba_point.y;
		//cloud_aux[i]->z = ba_point.z;


	}


///////


            Matx34d P;
            P(0,0) = R[i].at<double>(0,0); P(0,1) = R[i].at<double>(0,1); P(0,2) = R[i].at<double>(0,2); P(0,3) = T[i].at<double>(0);
            P(1,0) = R[i].at<double>(1,0); P(1,1) = R[i].at<double>(1,1); P(1,2) = R[i].at<double>(1,2); P(1,3) = T[i].at<double>(1);
            P(2,0) = R[i].at<double>(2,0); P(2,1) = R[i].at<double>(2,1); P(2,2) = R[i].at<double>(2,2); P(2,3) = T[i].at<double>(2);

            camera->P_mat = P;

        cam_matrix = cameraMatrix[i];
        dist_coeff =  distCoeffs[i];

        cout<<"Estimated D: "<<distCoeffs[i]<<endl;



    }

    catch(...)
    {
        cout<<"Something went wrong with BA. Maybe a wrong camera was estimated"<<endl;

	        	for (long i = 0; i <  cloud_aux.size(); i++) //global
		{
		    // if(cloud_aux[i]->idx !=-1 )
	      // {

		    Point3d ba_point = points[cloud_aux[i]->idx];
	        cloud_aux[i]->idx = -1;
			cloud_aux[i]->x = ba_point.x;
			cloud_aux[i]->y = ba_point.y;
			cloud_aux[i]->z = ba_point.z;

			//if(local_opt)
			cloud_aux[i]->can_ba = 0;

	      // }
		}
    }
}




bool sort_cloud(cloudPoint* c1, cloudPoint* c2)
{
	return c1->can_ba < c2->can_ba;
}

void adjust_local_ba_cvsba(vector<cloudPoint*> pointcloud, vector<Mat> &cam_matrix,vector<Mat> &dists, vector<imgMatches*> all_Pmats, int rows, int cols, bool local_opt, vector<int> idx_cams, bool not_final)
{

    vector<imgMatches*> Pmats;
    vector<imgMatches*> fixed_Pmats;
    //cv::Mat Kinv = cam_matrix.inv();
    vector<int> idx_cams2;


        for(int i=0;i<idx_cams.size();i++)
        {
             if(all_Pmats[idx_cams[i]]->triangulated)
             {
                 Pmats.push_back(all_Pmats[idx_cams[i]]);
                 idx_cams2.push_back(idx_cams[i]);
            }
        }




	long N = Pmats.size(), M = Count3DMeasurements(pointcloud), K=0 ;


    vector<Point3d> points;		// positions of points in global coordinate system (input and output)
    vector<cloudPoint*> cloud_aux;
	vector<vector<Point2d> > imagePoints; // projections of 3d points for every camera
	vector<vector<int> > visibility; // visibility of 3d points for every camera
	vector<Mat> cameraMatrix;	// intrinsic matrices of all cameras (input and output)
	vector<Mat> R;				// rotation matrices of all cameras (input and output)
	vector<Mat> T;				// translation vector of all cameras (input and output)
	vector<Mat> distCoeffs;		// distortion coefficients of all cameras (input and output)
    cv::TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 70, 1e-10);
    long ba_offset=0;






long cloud_idx=0;


////////////////////right//////////////

	//loading 3D points
/*	for (long i = 0, j=0; i < pointcloud.size();i++)
	{
      if(pointcloud[i]->is_in_cloud)
       {
        Point3d pt3d;

        pt3d.x = pointcloud[i]->x;
		pt3d.y = pointcloud[i]->y;
		pt3d.z = pointcloud[i]->z;
		points.push_back(pt3d);
        pointcloud[i]->idx = j;  j++;


       }
	}
*/

for (int i = 0; i < N; i++) //for all cameras
	{

         for(int j=0,k=0;j<Pmats[i]->pkeypoints.size();j++)
         {


            if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            {
             cloudPoint * cp = Pmats[i]->pkeypoints[j]->cloud_p;


               if(cp->idx ==-1)
                 { // give an index for it and add to the 3d points vector
                 	//int qtd_projs=0;

                 	//for(int k=0;k<cp->owners.size();k++)
                 	//	if(cp->owners[k]->img_owner->triangulated)
                 	//		qtd_projs++;

                    cloud_aux.push_back(cp);

                    cp->idx = 1;//cloud_idx++;

                 }

            }
        }
    }

std::sort(cloud_aux.begin(),cloud_aux.end(),sort_cloud);

for(long i=0;i<cloud_aux.size();i++)
{

	                Point3d pt3d;

                    pt3d.x = cloud_aux[i]->x;
                    pt3d.y = cloud_aux[i]->y;
                    pt3d.z = cloud_aux[i]->z;

                    points.push_back(pt3d);

                    cloud_aux[i]->idx = cloud_idx++;

                    if(cloud_aux[i]->can_ba==0 && local_opt)
                    	ba_offset++;
}


////////////////////end loading 3d points////////////////////////


	for (int i = 0; i < N; i++) //for all cameras
	{

		cameraMatrix.push_back(cam_matrix[idx_cams2[i]]);
		distCoeffs.push_back(dists[idx_cams2[i]]);//(cv::Mat_<double>(5,1) << 0, 0, 0, 0, 0));


		Matx34d P = Pmats[i]->P_mat;
		Mat_<double> KP1 = cam_matrix[idx_cams2[i]] * Mat(P); //Projection matrix

		Mat_<double> camR(3,3),camT(3,1);
		camR(0,0) = P(0,0); camR(0,1) = P(0,1); camR(0,2) = P(0,2); camT(0) = P(0,3);
		camR(1,0) = P(1,0); camR(1,1) = P(1,1); camR(1,2) = P(1,2); camT(1) = P(1,3);
		camR(2,0) = P(2,0); camR(2,1) = P(2,1); camR(2,2) = P(2,2); camT(2) = P(2,3);
		R.push_back(camR);
		T.push_back(camT);

         vector<Point2d> img_pts(points.size());
         vector<int> visib_cam(points.size(),0);

         for(int j=0,k=0;j<Pmats[i]->pkeypoints.size();j++)
         {


            if(Pmats[i]->pkeypoints[j]->cloud_p!=NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            {
                 cloudPoint * cp = Pmats[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                Point2d cvp (Pmats[i]->pkeypoints[j]->pt.x,Pmats[i]->pkeypoints[j]->pt.y); //adding points of this view



                img_pts[Pmats[i]->pkeypoints[j]->cloud_p->idx]=cvp;

                 //if(cvp.x >=0 && cvp.y>=0 && cvp.x<cols && cvp.y< rows)
                visib_cam[Pmats[i]->pkeypoints[j]->cloud_p->idx] = 1;

                 //cout<<"Predicted:" <<cvp0 << "Actual:" << cvp <<endl;
                 Pmats[i]->pkeypoints[j]->cloud_p->reproj_error+=norm(cvp0-cvp);

            }

         }

         imagePoints.push_back(img_pts);
         visibility.push_back(visib_cam);
	}


    cvsba::Sba sba;
    cvsba::Sba::Params param;

    if(not_final)
    {
    param.fixedDistortion=5;//3 //optimize k1 and k2
    param.fixedIntrinsics=5;//4 //optimize fx..
    param.iterations=70;
    }
    else
    {
    param.fixedDistortion=0; //optimize none
    param.fixedIntrinsics=4; //optimize none
    param.iterations=60;
    }


    param.verbose = false;
    param.minError = 1e-10;// 0.002; //1e-10
    param.type = sba.MOTIONSTRUCTURE;

    sba.setParams(param);


    try
    {

     //ncon(3dpoints), mcon(cameras)
     sba.run(ba_offset,0,points, imagePoints, visibility, cameraMatrix, R, T, distCoeffs);

     std::cout<<"Initial error="<<sba.getInitialReprjError()<<". "<<
                "Final error="<<sba.getFinalReprjError()<<std::endl;





	for (long i = 0; i <  cloud_aux.size(); i++) //global
	{
	    // if(cloud_aux[i]->idx !=-1 )
      // {

	    Point3d ba_point = points[cloud_aux[i]->idx];
        cloud_aux[i]->idx = -1;
		cloud_aux[i]->x = ba_point.x;
		cloud_aux[i]->y = ba_point.y;
		cloud_aux[i]->z = ba_point.z;

		//if(local_opt)
		cloud_aux[i]->can_ba = 0;

      // }
	}


///////

    for (int i = 0; i < N; i++)
        {
            Matx34d P;
            P(0,0) = R[i].at<double>(0,0); P(0,1) = R[i].at<double>(0,1); P(0,2) = R[i].at<double>(0,2); P(0,3) = T[i].at<double>(0);
            P(1,0) = R[i].at<double>(1,0); P(1,1) = R[i].at<double>(1,1); P(1,2) = R[i].at<double>(1,2); P(1,3) = T[i].at<double>(1);
            P(2,0) = R[i].at<double>(2,0); P(2,1) = R[i].at<double>(2,1); P(2,2) = R[i].at<double>(2,2); P(2,3) = T[i].at<double>(2);

            Pmats[i]->P_mat = P;

        cam_matrix[idx_cams2[i]] = cameraMatrix[i];
        dists[idx_cams2[i]] =  distCoeffs[i];
        }


    }

    catch(...)
    {
        cout<<"Something went wrong with BA. Maybe a wrong camera was estimated"<<endl;

	        	for (long i = 0; i <  cloud_aux.size(); i++) //global
		{
		    // if(cloud_aux[i]->idx !=-1 )
	      // {

		    Point3d ba_point = points[cloud_aux[i]->idx];
	        cloud_aux[i]->idx = -1;
			cloud_aux[i]->x = ba_point.x;
			cloud_aux[i]->y = ba_point.y;
			cloud_aux[i]->z = ba_point.z;

			//if(local_opt)
			cloud_aux[i]->can_ba = 0;

	      // }
		}
    }
}


void adjust_ba_pba(vector<cloudPoint*> pointcloud, Mat& cam_matrix, vector<imgMatches*> all_Pmats, int rows, int cols, ParallelBA::DeviceT device )
{
/*
    ParallelBA pba(device);
    long back_pts = 0;


    vector<imgMatches*> Pmats;
    //cv::Mat Kinv = cam_matrix.inv();



    for(int i=0;i<all_Pmats.size();i++)
      if(all_Pmats[i]->triangulated)
       Pmats.push_back(all_Pmats[i]);



	long N = Pmats.size(), M = Count3DMeasurements(pointcloud), K=0 ;


    /////////////////////////////////////////////////////////////////
    //CameraT, Point3D, Point2D are defined in src/pba/DataInterface.h
    vector<CameraT>        camera_data;    //camera (input/ouput)
    vector<Point3D>        point_data;     //3D point(iput/output)
    vector<Point2D>        measurements;   //measurment/projection vector
    vector<int>          camidx, ptidx;  //index of camera/point for each projection

    /////////////////////////////////////

    point_data.resize(M);

	//loading 3D points
	for (long int i = 0, j=0; i < pointcloud.size();i++)
	{
      if(pointcloud[i]->is_in_cloud)
       {
        point_data[j].SetPoint(pointcloud[i]->x,pointcloud[i]->y,pointcloud[i]->z);

        pointcloud[i]->idx = j;  j++;
    //  point_data[j-1].GetPoint(xyz2);
     // cout<<"La vai "<<xyz2[0] <<" / "<<pointcloud[i]->x <<" " <<xyz2[1] <<" / "<<pointcloud[i]->y <<" " << xyz2[2] <<" / "<<pointcloud[i]->z << endl;
       }
	}




    camera_data.resize(N); // allocate the camera data

	for (int i = 0; i < N; i++) //for all cameras
	{


       double f, q[9], c[3], d[2];
       f = cam_matrix.at<double>(0,0);


		Matx34d P = Pmats[i]->P_mat;
		Mat_<double> KP1 = cam_matrix * Mat(P); //Projection matrix

		q[0] = P(0,0); q[1] = P(0,1); q[2] = P(0,2); c[0] = P(0,3);
		q[3] = P(1,0); q[4] = P(1,1); q[5] = P(1,2); c[1] = P(1,3);
		q[6] = P(2,0); q[7] = P(2,1); q[8] = P(2,2); c[2] = P(2,3);

            camera_data[i].SetFocalLength(f);
            camera_data[i].SetMatrixRotation(q);
            camera_data[i].SetTranslation(c);

            camera_data[i].SetProjectionDistortion(0);


         for(int j=0,k=0;j<Pmats[i]->pkeypoints.size();j++)
         {
            if(Pmats[i]->pkeypoints[j]->cloud_p != NULL && Pmats[i]->pkeypoints[j]->cloud_p->is_in_cloud)
            Pmats[i]->pkeypoints[j]->cam_idx = i;
         }

	}

	for(long i=0;i<pointcloud.size();i++)
	 if(pointcloud[i]->is_in_cloud)
        for(int j=0;j<pointcloud[i]->owners.size();j++)
            {

                if(pointcloud[i]->owners[j]->cam_idx!=-1)
                {

                measurements.push_back(Point2D(pointcloud[i]->owners[j]->pt.x - cols/2.0, -(-pointcloud[i]->owners[j]->pt.y + rows/2.0)));
                camidx.push_back(pointcloud[i]->owners[j]->cam_idx); //
                pointcloud[i]->owners[j]->cam_idx = -1;
                ptidx.push_back(pointcloud[i]->idx);
                }


            }

    ////////////////////////////////////////////////////////////////
    pba.SetCameraData(camera_data.size(),  &camera_data[0]);                        //set camera parameters
    pba.SetPointData(point_data.size(), &point_data[0]);                            //set 3D point data
    pba.SetProjection(measurements.size(), &measurements[0], &ptidx[0], &camidx[0]);//set the projections
    pba.SetFixedIntrinsics(true);


    pba.RunBundleAdjustment();    //run bundle adjustment, and camera_data/point_data will be modified

	for (long i = 0; i <  pointcloud.size(); i++) //global
	{
	     if(pointcloud[i]->is_in_cloud)
       {
	    Point3D ba_point = point_data[pointcloud[i]->idx];
	    double vet[3];
	    ba_point.GetPoint(vet);

		pointcloud[i]->x = vet[0];
		pointcloud[i]->y = vet[1];
		pointcloud[i]->z = vet[2];

       }
	}




///////

    for (int i = 0; i < N; i++)
        {
            Matx34d P;
            double r[9];
            double t[3];
            double f = camera_data[i].GetFocalLength();

            camera_data[i].GetMatrixRotation(r);
            camera_data[i].GetTranslation(t);

            P(0,0) = r[0]; P(0,1) = r[1]; P(0,2) = r[2]; P(0,3) = t[0];
            P(1,0) = r[3]; P(1,1) = r[4]; P(1,2) = r[5]; P(1,3) = t[1];
            P(2,0) = r[6]; P(2,1) = r[7]; P(2,2) = r[8]; P(2,3) = t[2];

            Pmats[i]->P_mat = P;

            //cam_matrix.at<double>(0,0) = f;
            //cam_matrix.at<double>(1,1) = f;

        }
*/

}


