// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


//Sites de referencia:
//http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html
//https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/c/detectors_sample.cpp?rev=3052
//---------------------------------------------------------------------------------------------
//#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_removal.h>

#include<utility>
#include<string.h>
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<queue>
#include<map>
#include<set>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <unistd.h>
//#include <sys/time.h>

//#include "codigos/Utilitarios.h"
#include "codigos/ceres_BA.cpp"
#include "codigos/BundleAdjustment.h" //
#include "codigos/VocabTree.h"
#include "codigos/AC_Ransac.h"


//#include <pcl/visualization/cloud_viewer.h>

//reconstruction
/*#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/passthrough.h>
*/




#define EPSILON 0.0001

#define GLOBAL_BA true //run full ba in the end

#define LOCAL_BA true //perform local BA in the reconstruction progress

#define ACONTRARIO true //use the a contrario scheme for RANSAC

#define GBA_FACTOR 2.0 //1.33, 1.5, 1.7

#define UPPER_BOUND 40.0

#define FILTER_POINTS true

using namespace cv;
using namespace std;

//

cv::Mat K; //intrinsic calib
cv::Mat D;
vector<cv::Mat> Kvec; //intrinsic for each cams
vector<cv::Mat> Dvec; // distortion for each cam
cv::Mat distortion; //distortion coeffs of lens
int c_img=0;
int rectf_count=0;
int pair_count=0;
int pnp_count=0;
long ba_counter = 1;
int next_ba;
pair<double,long> confidence;



struct timeval start, end;
long long mtime, seconds, useconds;

struct pairMatch
{
    vector<long> idx_cam1, idx_cam2;
    vector<Point2f> cam1, cam2;
    Mat F_mat;
    bool triangulated;
    bool h_tested;
    int homog_inliers;
};


#include "codigos/thread_Match.h"
#include "codigos/parallel_Features.h"
#include "codigos/input_output.h"
#include "codigos/graph_utils.h"


void remove_outliersfromcloud(vector<cloudPoint*> Cloud);
void reconstruct_surface(vector<cloudPoint*> Cloud);

cv::Mat find_ray(Matx34d P1,  Point2f n_x1);
cv::Mat find_ray_3d(Matx34d P1,  Mat X); //Normalized P_mat and 3D point position

double angle_between_vec(Mat v1, Mat v2);


Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                   Matx34d P,       //camera 1 matrix
                   Point3d u1,      //homogenous image point in 2nd camera
                   Matx34d P1       //camera 2 matrix
                                   );

                                   bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2);

	Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
											Matx34d P,			//camera 1 matrix
											Point3d u1,			//homogenous image point in 2nd camera
											Matx34d P1			//camera 2 matrix
											);



bool CheckCoherentRotation(cv::Mat_<double>& R) {


	if(fabsf(determinant(R))-1.0 > 1e-07) {
		//cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}



bool TestTriangulation(vector<Point3d> pcloud_pt3d,  Matx34d P)
{

	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	vector<uchar> status;

	Matx44d P4x4 = Matx44d::eye();

	for(int i=0;i<12;i++)
	P4x4.val[i] = P.val[i];

	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

	status.resize(pcloud_pt3d.size(),0);
	for (int i=0; i<pcloud_pt3d.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud_pt3d.size());
	//cout << count << "/" << pcloud_pt3d.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75) // || percentage > 0.985)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
         cv::Mat_<double> cldm(pcloud_pt3d.size(),3);
        for(unsigned int i=0;i<pcloud_pt3d.size();i++) {
            cldm.row(i)(0) = pcloud_pt3d[i].x;
            cldm.row(i)(1) = pcloud_pt3d[i].y;
            cldm.row(i)(2) = pcloud_pt3d[i].z;
        }
        cv::Mat_<double> mean;
        cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

        double p_to_plane_thresh = pca.eigenvalues.at<double>(2);
        int num_inliers = 0;
        cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
        cv::Vec3d x0 = pca.mean;

        for (int i=0; i<pcloud_pt3d.size(); i++)
         {
            Vec3d w = Vec3d(pcloud_pt3d[i]) - x0;
            double D = fabs(nrm.dot(w));
            if(D < p_to_plane_thresh) num_inliers++;
          }

        cout << num_inliers << "/" << pcloud_pt3d.size() << " are coplanar" << endl;
        if((double)num_inliers / (double)(pcloud_pt3d.size()) > 0.9)
            return false;
	}

	return true;
}



double triangulatePoints_opencv(cv::Matx34d P1, cv::Matx34d P, vector<Point2f> cam1, vector<Point2f> cam2, vector<Point3d> &pcloud, vector<double> reproj_error)
{
    vector<Point2d> cam_1, cam_2;
    Mat triangulated_3d;


    for (int i=0; i<cam1.size(); i++)
	{
        Point3f pt3d;
		Point2f kp = cam1[i];
		Point3d u(kp.x,kp.y,1.0);

		Mat_<double> um = K.inv() * Mat_<double>(u);
		u.x = um(0); u.y = um(1); u.z = um(2);

		Point2f kp1 = cam2[i];
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = K.inv() * Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Point2d pc1 = Point2d(u.x/u.z,u.y/u.z);
		Point2d pc2 = Point2d(u1.x/u1.z, u1.y/u1.z);

		cam_1.push_back(pc1);
		cam_2.push_back(pc2);


	}

    triangulatePoints(P,P1,cam_1,cam_2,triangulated_3d);

    for(int i=0;i<triangulated_3d.cols;i++)
    {
        double h = triangulated_3d.at<double>(3,i);
        Point3d p = Point3d(triangulated_3d.at<double>(0,i)/h,triangulated_3d.at<double>(1,i)/h,triangulated_3d.at<double>(2,i)/h);
        pcloud.push_back(p);
    }

    return 0;
}


double TriangulatePoints(cv::Matx34d P1, cv::Matx34d P, vector<Point2f> cam1, vector<Point2f> cam2, vector<Point3d> &pcloud, vector<double> &reproj_error)
{


    cv::Mat Kinv = K.inv();
    pcloud.clear();
    reproj_error.clear();

    //transforming to NCC



   /* for(int i=0;i<cam1_.size();i++)
    {
        Mat  p = (cv::Mat_<float>(1,3) << cam1_[i].x,cam1_[i].y,1);
        Mat  p2 = (cv::Mat_<float>(1,3) << cam2_[i].x,cam2_[i].y,1);

        p = K.inv()*p;
        p2= K.inv()*p2;

        Point2f p_1 = Point2f(p.at<float>(0,0),p.at<float>(0,1));
        Point2f p_2 = Point2f(p2.at<float>(0,0),p2.at<float>(0,1));
        cam1.push_back(p_1);
        cam2.push_back(p_2);

    }*/

    //vector<double> reproj_error;

	Mat_<double> KP1 = K * Mat(P1);
	Mat_<double> KP2 = K* Mat(P);

	for (int i=0; i<cam1.size(); i++)
	{
        Point3f pt3d;
		Point2f kp = cam1[i];
		Point3d u(kp.x,kp.y,1.0);

		Mat_<double> um = Kinv * Mat_<double>(u);
		u.x = um(0); u.y = um(1); u.z = um(2);

		Point2f kp1 = cam2[i];
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);

        pt3d.x = X(0);
        pt3d.y = X(1);
        pt3d.z = X(2);

        pcloud.push_back(pt3d);

        Mat_<double> xPt_img = KP1 * X;				//reproject
		Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
		double reprj_err = norm(xPt_img_-kp1);

        Mat_<double> xPt_img2 = KP2 * X;				//reproject
		Point2f xPt_img_2(xPt_img2(0)/xPt_img2(2),xPt_img2(1)/xPt_img2(2));
		double reprj_err2 = norm(xPt_img_2-kp);


        reproj_error.push_back(max(reprj_err,reprj_err2));
        }

        //Scalar mse = mean(reproj_error);
        std::sort(reproj_error.begin(),reproj_error.end());

        //return mse[0];
        return reproj_error[reproj_error.size()/2];

}

double TriangulatePointsRectified(cv::Matx34d P1, cv::Matx34d P, vector<Point2f> cam1, vector<Point2f> cam2, int idxc1, int idxc2, vector<Point3d> &pcloud, vector<double> &reproj_error, vector<bool> &inliers)
{


    cv::Mat Kinv = K.inv();//
    pcloud.clear();
    reproj_error.clear();
    inliers.clear();


	Mat_<double> KP1 = Kvec[idxc2] * Mat(P1);
	Mat_<double> KP2 = Kvec[idxc1]* Mat(P);

	//undistort points

	openMVG::sfm::undistort_points(cam1, Kvec[idxc1], Dvec[idxc1]);
	openMVG::sfm::undistort_points(cam2, Kvec[idxc2], Dvec[idxc2]);

    int cont_outs=0;
    int cont_reprjs=0;

	for (int i=0; i<cam1.size(); i++)
	{
        Point3f pt3d;


        Point2f kp = cam1[i];
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kvec[idxc1].inv() * Mat_<double>(u); //normalized
		u.x = um(0); u.y = um(1); u.z = um(2);


        Point2f kp1 = cam2[i];
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kvec[idxc2].inv() * Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Point2f pt_n1 = Point2f(u.x,u.y);
		Point2f pt_n2 = Point2f(u1.x,u1.y);

		//cout<<"Angle = "<<angle_between_vec(find_ray(P,pt_n1), find_ray(P1,pt_n2))<<endl; getchar();

		Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);

        pt3d.x = X(0);
        pt3d.y = X(1);
        pt3d.z = X(2);

        pcloud.push_back(pt3d);

        Mat_<double> xPt_img = KP1 * X;				//reproject
		Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));
		double reprj_err = norm(xPt_img_-kp1);

        Mat_<double> xPt_img2 = KP2 * X;				//reproject
		Point2f xPt_img_2(xPt_img2(0)/xPt_img2(2),xPt_img2(1)/xPt_img2(2));
		double reprj_err2 = norm(xPt_img_2-kp);

		double max_rerr = max(reprj_err,reprj_err2);

		Mat_<double> pt3d_n1 = Mat(P1)*X;
		Mat_<double> pt3d_n2 = Mat(P)*X;


        reproj_error.push_back(max_rerr);
        if(max_rerr > 2.0)
        	cont_reprjs++;

        if(angle_between_vec(find_ray_3d(P,X), find_ray_3d(P1,X)) < 2.0 || pt3d_n1(2) <= 0.0 || pt3d_n2(2) <= 0.0) // validate by triangulation angle and verify if they are in the back of cameras
        	{inliers.push_back(false); cont_outs++;}
        else
        	inliers.push_back(true);


        }

        /*cv::Mat_<double> ray1(3,1), ray2(3,1);
        ray1(0) = 413; ray1(2) =585; ray1(3) = 0;
        ray2(0) = 721; ray2(2) = 333; ray2(3) = 0;
        cout<<angle_between_vec(ray1,ray2)<<endl;getchar();
		*/

        //Scalar mse = mean(reproj_error);
        std::sort(reproj_error.begin(),reproj_error.end());
        cout<<"Count triangulation outliers = "<<cont_outs <<" in a total of "<<reproj_error.size()<<endl;//". % Points w/ large error "<<100.0*(cont_reprjs/(double)reproj_error.size())<<endl;

        //return mse[0];
       // cout<<"triangulation median: "<<reproj_error[reproj_error.size()*0.9]<<endl; getchar();
        return reproj_error[reproj_error.size()*0.5];

}

bool find_P1(cv::Matx34d &P1, cv::Mat F, vector<Point2f> cam1, vector<Point2f> cam2, vector<double> &reproj_error, int idx_c1, int idx_c2, vector<bool> &inliers)
{

       //Essential matrix: compute then extract cameras [R|t]
		Mat_<double> E = F;//Kvec[idx_c1].t() * F * Kvec[idx_c2]; //according to HZ (9.12)
		vector<Point3d> pcloud;
		vector<Point3d> pcloud2;
		//vector<bool> inliers;

		 cv::Matx34d P(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0);



    if(fabsf(determinant(E)) > 1e-07)
    {
			//cout << "det(E) != 0 : " << determinant(E) << "\n";
//			P1 = 0;
			//return false;
		}

		Mat_<double> R1(3,3);
		Mat_<double> R2(3,3);
		Mat_<double> t1(1,3);
		Mat_<double> t2(1,3);

		//decompose E to P' , HZ (9.19)

			if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

			if(determinant(R1)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				//cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
				E = -E;
				DecomposeEtoRandT(E,R1,R2,t1,t2);
			}
			if (!CheckCoherentRotation(R1)) {
				cout << "resulting rotation is not coherent\n";
//				P1 = 0;
				return false;
			}

			P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
			//cout << "Testing 1 " << endl << Mat(P1) << endl;


			double reproj_error1 = TriangulatePointsRectified(P1,P,cam1,cam2,idx_c1,idx_c2,pcloud,reproj_error,inliers);
			double reproj_error2 = TriangulatePointsRectified(P,P1,cam2,cam1,idx_c2,idx_c1,pcloud2,reproj_error,inliers);
			 //cout<<"Reproj error: "<<reproj_error1<<endl;

              //cout<<" reproj error 1:"<<reproj_error1<<endl;
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
		if (!TestTriangulation(pcloud,P1)  || !TestTriangulation(pcloud2,P)|| reproj_error1 > 100.0 || reproj_error2>100.0)
			{
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				//cout << "Testing 2 "<< endl << Mat(P1) << endl;


				 reproj_error1 = TriangulatePointsRectified(P1,P,cam1,cam2,idx_c1,idx_c2,pcloud,reproj_error,inliers);
                 reproj_error2 = TriangulatePointsRectified(P,P1,cam2,cam1,idx_c2,idx_c1,pcloud2,reproj_error,inliers);
                  //cout<<"Reproj error: "<<reproj_error1<<endl;

				if (!TestTriangulation(pcloud,P1)  || !TestTriangulation(pcloud2,P)|| reproj_error1 > 100.0 || reproj_error2>100.0){
					if (!CheckCoherentRotation(R2)) {
						cout << "resulting rotation is not coherent\n";
						P1 = 0;
						return false;
					}

					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					//cout << "Testing 3 "<< endl << Mat(P1) << endl;


					  reproj_error1 = TriangulatePointsRectified(P1,P,cam1,cam2,idx_c1,idx_c2,pcloud,reproj_error,inliers);
                 reproj_error2 = TriangulatePointsRectified(P,P1,cam2,cam1,idx_c2,idx_c1,pcloud2,reproj_error,inliers);
                  //cout<<"Reproj error: "<<reproj_error1<<endl;

					if (!TestTriangulation(pcloud,P1)  || !TestTriangulation(pcloud2,P)|| reproj_error1 > 100.0 || reproj_error2>100.0){
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						//cout << "Testing 4 "<< endl << Mat(P1) << endl;


						 reproj_error1 = TriangulatePointsRectified(P1,P,cam1,cam2,idx_c1,idx_c2,pcloud,reproj_error,inliers);
                        reproj_error2 = TriangulatePointsRectified(P,P1,cam2,cam1,idx_c2,idx_c1,pcloud2,reproj_error,inliers);
                        //cout<<"Reproj error: "<<reproj_error1<<endl;

						if (!TestTriangulation(pcloud,P1)  || !TestTriangulation(pcloud2,P)|| reproj_error1 > 100.0 || reproj_error2>100.0) {
							//cout << "nope" << endl;
							return false;
						}
					}
				}
			}







    return true;
}

double euclidean_distance_latlong(double lat1, double long1, double lat2, double long2)
{
    Point2d p1, p2;

    p1.x = lat1; p1.y = long1;
    p2.x = lat2; p2.y = long2;

    return norm(p1-p2);

}

double toRadians(double v)
{
    return v*(M_PI/180.0);

}

double distFrom(double lat1, double lng1, double lat2, double lng2)
{
    double earthRadius = 6378137; //meters
    double dLat =toRadians(lat2-lat1);
    double dLng =toRadians(lng2-lng1);
    double a = sin(dLat/2) * sin(dLat/2) +
               cos(toRadians(lat1)) * cos(toRadians(lat2)) *
               sin(dLng/2) * sin(dLng/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    double dist = (double) (earthRadius * c);

    return abs(dist);
}


Mat getK(double focal_length, double sensor_size_x, double sensor_size_y, int img_width, int img_height)
{


cv::Mat estK;
double w_certo, h_certo;

if(img_width>img_height)
{
	w_certo = img_width; h_certo = img_height;
}
else
{ w_certo = img_height; h_certo = img_width; }

    if(focal_length==0)
    {
        focal_length = 4.3; // assuming a generic FoV
        sensor_size_x = 6.16;
        estK =  (cv::Mat_<double>(3,3) << w_certo*1.0,0.0, (img_width/2.0),
                                                0.0, w_certo*1.0,(img_height/2.0),
                                                0.0,                0.0,                           1.0 	 );


           cout<<"Using est. calibration matrix: " <<endl << estK<<endl <<"Press key..."; getchar();

     }

    else{
        estK =  (cv::Mat_<double>(3,3) << (focal_length*(double)w_certo)/sensor_size_x,0.0, img_width/2.0 - 1.0,
                                                0.0, (focal_length*(double)h_certo)/sensor_size_y, img_height/2.0 - 1.0,
                                                0.0,                0.0,                           1.0 	 );


        }

 return estK;

}


void generate_I_Graph(Mat &G, vector<imgMatches*> imgs_features)
{
	if(true)
	{
    vector<vector<int> > all_queries;
    all_queries = vocabulary_tree_prune(imgs_features); //returns the highest k scores from the vocabulary tree query for each image in imgs_features.

    for(int i=0;i<all_queries.size();i++)
        for(int j=0;j<all_queries[i].size();j++)
            G.at<double>(i,all_queries[i][j]) = 1;
    }
    else
    {
    	cout<<"Warning: Generating full-connected graph."<<endl;
    	for(int i=0;i<G.rows;i++)
    		for(int j=i+1;j<G.cols;j++)
    			G.at<double>(i,j) = 1;
    }


            //cout<<"Initial vocab graph:"<<G<<endl;


}

void generate_D_Graph(Mat &G, vector<Point2d> gps_data) //calculates distance in meters between all imgs
{
    long cont_m=0;
    //initializing
//cout<<"Before prune: "<<G<<endl;
    for(int i=0;i<G.rows && i < G.cols;i++) //zeroing the main diagonal to avoid corresponding img to itself
    	G.at<double>(i,i)= 0;

    for(int i=0;i<G.rows;i++)
     for(int j=i+1;j<G.cols;j++)
     {

       //  if(gps_data[i].x == 0 || gps_data[i].y == 0 || gps_data[j].x == 0 || gps_data[j].y == 0)
        // G.at<double>(i,j) = -1;
        // else
       
        if(G.at<double>(i,j)==0) //check if vocab_tree considered this edge
        {

               // G.at<double>(i,j) = 1;
               // G.at<double>(j,i) = 0;

            if(!(gps_data[i].x == 0 || gps_data[i].y == 0 || gps_data[j].x == 0 || gps_data[j].y == 0))
            {
                  G.at<double>(i,j) = distFrom(gps_data[i].x,gps_data[i].y,gps_data[j].x,gps_data[j].y);

            }

        }

         //cout<<G.at<double>(i,j)<<endl; getchar();
        // if(G.at<double>(i,j)==-1)
        // cont_m++;
     }

     cout<<cont_m<<endl; //getchar();
}

void cut_D_Graph(Mat &G, double meters_threshold) // cut edges based on a threshold in meters
{
	
    for(int i=0;i<G.rows;i++)
     for(int j=i+1;j<G.cols;j++)
     {
        if(G.at<double>(i,j)!=0)
        {
            if( G.at<double>(i,j) > meters_threshold)
            G.at<double>(i,j) = 0;
            else
            {

             pair_count++;
             G.at<double>(i,j) = 1;
            }
        }
     }
}



bool sort_matches(DMatch p1, DMatch p2)
{
    return (p1.distance < p2.distance);
}

void get_best_matches(vector<DMatch> matches, vector<DMatch> &good_matches, double percentile)
{
    good_matches.clear();
    vector<DMatch> ordered_matches;

    for(unsigned int i=0;i<matches.size(); i++)
        ordered_matches.push_back(matches[i]);

    std::sort(ordered_matches.begin(),ordered_matches.end(),sort_matches);

    for(unsigned int i=0;i< (unsigned int) ( (double)matches.size() * percentile); i++)
        good_matches.push_back(ordered_matches[i]);

}


double match_pair(imgMatches *im1, imgMatches *im2, Mat &F_mat, int qualDescritor, vector<Point2f> &cam_1, vector<Point2f> &cam_2, vector<long> &idx_cam1, vector<long> &idx_cam2, vector<cloudPoint*> &Cloud, Mat img1, Mat img2, int &homog_inliers,bool use_flann, double good_matches_percentile, double inliers_threshold, int id_cam1, int id_cam2) //returns qtd de inliers
{

    //calculating matching
    //

    int qualnorm;
    bool write_img= true;
    bool use_fast_matching = false;
    int MAX_KEYPOINTS = 500;


    if(qualDescritor<2)
    qualnorm = NORM_L2;
    else
    qualnorm = NORM_HAMMING;

    BFMatcher matcher(qualnorm,true);
    FlannBasedMatcher fnn_matcher;

    vector<DMatch> matches;

    /////////////////////////////////////////////////



    if(use_fast_matching)
    {
        int inlier_c=0;

        cv::Mat im2k, im1k;
        vector<Point2f> cam1t,cam2t;


        /*im2k.create(MAX_KEYPOINTS,im2->descriptors_img.cols,CV_32F);
        im1k.create(MAX_KEYPOINTS,im2->descriptors_img.cols,CV_32F);

        for(int i=0;i<MAX_KEYPOINTS;i++)
        {
           for(int j=0;j<im2->descriptors_img.cols;j++)
            {
                im2k.at<float>(i,j)=im2->descriptors_img.at<float>(i,j);
                im1k.at<float>(i,j)=im1->descriptors_img.at<float>(i,j);
            }

        }
        */
        if(im2->descriptors_img.rows<10 || im1->descriptors_img.rows<10)
            return 0;

        for(int i=0;i< im2->descriptors_img.rows && i< im1->descriptors_img.rows && i < MAX_KEYPOINTS;i++)
        {
        im2k.push_back(im2->descriptors_img.row(i));
        im1k.push_back(im1->descriptors_img.row(i));
        }

        matcher.match(im2k, im1k, matches);

        for(unsigned int i = 0; i < matches.size(); i++ )
        {
          cam1t.push_back( im1->keypoints_img[ matches[i].trainIdx ].pt );
          cam2t.push_back( im2->keypoints_img[ matches[i].queryIdx ].pt );
        }


    double threshold;
    std::vector<uchar> inlierst(cam1t.size(),0);

        Mat test_mat = findFundamentalMat(cam1t, cam2t, FM_RANSAC, 0.00320*(double)im1->cols, 0.999,inlierst);//0.00120
        // Mat test_mat = openMVG::sfm::FindFundamentalMat_AC_RANSAC(cam1t, cam2t,Kvec[id_cam1], Kvec[id_cam2], im1->cols, im1->rows, im2->cols, im2->rows, threshold, inlierst);

    std::vector<uchar>::const_iterator
    itInt= inlierst.begin();
    std::vector<cv::DMatch>::const_iterator
    itMt= matches.begin();

    // for all matches
    for ( ;itInt!= inlierst.end(); ++itInt, ++itMt)
    {
        if (*itInt) //  if inlier
        { // it is a valid match
            inlier_c++;

        }
    }

    if(inlier_c<0.15*matches.size())
    {
        F_mat = (Mat_<double>(3,3)<<0,0,0,0,0,0,0,0,0);
      //  cout<<"skipping match...."<<c_img <<" "<<inlier_c <<endl;
        c_img++;
        return 0;
    }
        cout<<"inlier pair ratio = "<<(inlier_c/(double)matches.size())*100<<"% "<<endl;
         matches.clear();


    }



    //////////////////////////////////////////////////


    //FlannBasedMatcher fnn_matcher2(new flann::LshIndexParams(20,10,2));
       vector<vector<DMatch> > k_matches;



	if(use_flann)
	{    if(qualnorm == NORM_L2)
	     fnn_matcher.knnMatch(im2->descriptors_img, im1->descriptors_img,k_matches,2,Mat(),false);
	     else
	     ;//fnn_matcher2.match(im2->descriptors_img, im1->descriptors_img, matches);
	}
	else
	matcher.match(im2->descriptors_img, im1->descriptors_img, matches);
    //cout<<matches.size()<<endl;
    //desenha matches
    vector<DMatch> good_matches;

    double min_dist=9999, max_dist=0;

     /*for( int i = 0; i < matches.size(); i++ )
      {
          double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }
     */

    //for(unsigned int i=0;i<matches.size();i++)
    //    good_matches.push_back(matches[i]);

    //getting the best % keypoint matches
    //get_best_matches(matches,good_matches,good_matches_percentile); //20
      //Criterion ratio test
      for(int i=0;i<k_matches.size();i++)
      {
      	if(k_matches[i][0].distance < 0.8*k_matches[i][1].distance)
      		good_matches.push_back(k_matches[i][0]);
      }

     //waitKey(0);
    Mat img_matches;
    if(write_img)
    {
    drawMatches(img2, im2->keypoints_img, img1, im1->keypoints_img,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }
                 //waitKey(0);

    //imshow("Good Matches", img_matches);


    string path =  "img_mina/result/matches_ransac";
    char buffer [33];
    sprintf(buffer," %d-%d ",c_img+1,pair_count);
    path+=buffer;
    path+=".jpg";
    cout<<"PATH "<<path<<endl;

	//imwrite( path, img_matches );




    vector<Point2f> cam1;
    vector<Point2f> cam2;

    vector<Point2f> n_cam1;
    vector<Point2f> n_cam2;
  //  Mat points3D;
    std::vector<uchar> inliers(cam1.size(),0);
    std::vector<uchar> inliers_homography(cam1.size(),0);



       for(unsigned int i = 0; i < good_matches.size(); i++ )
    {

      cam1.push_back( im1->keypoints_img[ good_matches[i].trainIdx ].pt );
	  cam2.push_back( im2->keypoints_img[ good_matches[i].queryIdx ].pt );

    }



 //  Mat fmat1 =  findFundamentalMat(cam1, cam1, FM_RANSAC, 2, 0.99);
   //Mat fmat1 = (cv::Mat_<double>(3,3) <<	1,	0	,		0,
	//											0,			1,	0,
	//											0,			0,			1); // Fundamental mat image 1

    double minVal, maxVal;
    minMaxIdx(cam1,&minVal,&maxVal);
    double threshold;


    Mat fmat2 = cv::findEssentialMat(cam1, cam2, Kvec[id_cam1].at<double>(0,0),Point2d(Kvec[id_cam1].at<double>(0,2),Kvec[id_cam1].at<double>(1,2)),cv::RANSAC, 0.999,(0.00220*(double)im1->cols),inliers);
 // Mat fmat2 = findFundamentalMat(cam1, cam2, FM_RANSAC, (0.00220*(double)im1->cols), 0.999,inliers); //0.00120
   // Mat fmat2 = openMVG::sfm::FindFundamentalMat_AC_RANSAC(cam1, cam2,Kvec[id_cam1], Kvec[id_cam2], im1->cols, im1->rows, im2->cols, im2->rows, threshold, inliers);
    cout<<"Found AC threshold: "<<threshold<<endl<<endl;

   Mat mat_homog = findHomography(cam1, cam2, FM_RANSAC,(0.00350*(double)im1->cols),inliers_homography);

   homog_inliers = 0;

    std::vector<uchar>::const_iterator itIn_h= inliers_homography.begin();


    // number of inliers homography
    for ( ;itIn_h!= inliers_homography.end(); ++itIn_h)
    {
        if (*itIn_h) //  if inlier
        { // it is a valid match
           homog_inliers++;

        }
    }

    //cout<<"eh "<<countNonZero(inliers)<<endl;
 //good_matches.clear();

vector<DMatch> good_matches_RANSAC;

   std::vector<uchar>::const_iterator
    itIn= inliers.begin();
    std::vector<cv::DMatch>::const_iterator
    itM= good_matches.begin();

    // for all matches
    for ( ;itIn!= inliers.end(); ++itIn, ++itM)
    {
        if (*itIn) //  if inlier
        { // it is a valid match
            good_matches_RANSAC.push_back(*itM);

        }
    }

        if(write_img)
        {


        drawMatches(img2, im2->keypoints_img, img1, im1->keypoints_img,
                good_matches_RANSAC, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
               // waitKey(0);
        }


  //  imshow("Good Matches 2", img_matches);
   string path2 =  "/local/Datasets/icex_small/images/result/matches/good_matches_ransac_";
    char buffer2 [33];
    sprintf(buffer2,"%d",c_img++);
    path2+=buffer2;
    path2+=".jpg";
    //cout<<"PATH "<<path<<endl;
	cv::resize(img_matches,img_matches,Size(),0.5,0.5);
    if(write_img)
    imwrite(path2, img_matches);



cam1.clear();
cam2.clear();

   if(good_matches_RANSAC.size() > inliers_threshold) // dont create a point cloud for outlier pair, cuz it doesn't makes sense.
   {
    for( int i = 0; i < good_matches_RANSAC.size(); i++ )
    {
      cam1.push_back( im1->keypoints_img[ good_matches_RANSAC[i].trainIdx ].pt );
	  cam2.push_back( im2->keypoints_img[ good_matches_RANSAC[i].queryIdx ].pt );

	  idx_cam1.push_back(good_matches_RANSAC[i].trainIdx);
	  idx_cam2.push_back(good_matches_RANSAC[i].queryIdx);



	  //adding corresponding points into a common point in cloud
	  if(im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p ==NULL && im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p ==NULL ) //not set yet
 	  {
 	      cloudPoint *cp = new cloudPoint;
 	      cp->is_in_cloud = false;
 	      cp->idx = -1;
 	      cp->x = cp->y = cp->z = 0;
 	      cp->reproj_error = 0;
 	      cp->can_ba = 1;

 	      cp->owners.push_back(im1->pkeypoints[good_matches_RANSAC[i].trainIdx]);
 	      cp->owners.push_back(im2->pkeypoints[good_matches_RANSAC[i].queryIdx]);

 	      im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p = cp;
 	      im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p = cp;


 	      Cloud.push_back(cp);

 	  }
 	  else if(im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p ==NULL && im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p != NULL )
 	  {
 	       im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p = im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p;
 	       im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p->owners.push_back(im1->pkeypoints[good_matches_RANSAC[i].trainIdx]);
 	  }
 	  else if(im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p != NULL && im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p == NULL )
 	  {
            im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p = im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p;
            im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p->owners.push_back(im2->pkeypoints[good_matches_RANSAC[i].queryIdx]);
 	  }
 	  else if( im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p != im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p)
 	  { //merge 2 with 1

        cloudPoint *c1, *c2;

        c1 = im1->pkeypoints[good_matches_RANSAC[i].trainIdx]->cloud_p;
        c2 = im2->pkeypoints[good_matches_RANSAC[i].queryIdx]->cloud_p;

 	      for(int j=0;j < c2->owners.size();j++)
 	      {
            c1->owners.push_back(c2->owners[j]);
            c2->owners[j]->cloud_p = c1;

 	      }



 	     for(int k=0;k<Cloud.size();k++)
 	      {

          	 if(Cloud[k] == c2)
 	       { Cloud[k]->owners.clear(); delete(Cloud[k]); Cloud.erase(Cloud.begin()+k); break;}

 	      }


 	  }

    }
   }

 //testing optima points
    if(fmat2.rows>0 && fmat2.cols>0 && cam1.size()>0 && cam2.size()>0)
    {

    //correctMatches(fmat2, cam1, cam2, cam_1, cam_2);
    cam_1 = cam1;
    cam_2 = cam2;

    F_mat = fmat2;
    }
    else
    {
        F_mat = (Mat_<double>(3,3)<<0,0,0,0,0,0,0,0,0);
    }
    return cam1.size();
}

/*
void show_pointCloud(vector<cloudPoint*> Cloud)
{
    vector<Point3d> pcloud;
    vector<Vec3b> pcrgb;

	for (long i=0; i<Cloud.size(); i++)
	{
      if(Cloud[i]->is_in_cloud)
      {


        Point3d pt3d;
		//Point2f kp = cam1[i];
		//Point3d u(kp.x,kp.y,1.0);
		Vec3b urgb (Cloud[i]->owners[0]->R, Cloud[i]->owners[0]->G,Cloud[i]->owners[0]->B);

		//Mat_<double> um = Kinv * Mat_<double>(u);
		//u.x = um(0); u.y = um(1); u.z = um(2);

		//Point2f kp1 = cam2[i];
		//Point3d u1(kp1.x,kp1.y,1.0);
		//Mat_<double> um1 = Kinv * Mat_<double>(u1);
		//u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		//Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1); //Triangulating points between the 2 initial views

        pt3d.x = Cloud[i]->x;
        pt3d.y = Cloud[i]->y;
        pt3d.z = Cloud[i]->z;

        pcloud.push_back(pt3d);
        pcrgb.push_back(urgb);

      }

    }


///////////////////////POINT CLOUD VISUALIZATION PCL/////////////////////////////////



pcl::visualization::CloudViewer viewer ("Sparse DEM");
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr mycloud(new pcl::PointCloud<pcl::PointXYZRGB>);
       // pcl::PointCloud<pcl::PointXYZ>::Ptr simplecloud(new pcl::PointCloud<pcl::PointXYZ>);

       //populating cloud

        for(long i=0;i<pcloud.size();i++)
        {
          //  pcl::PointXYZRGB o;
             pcl::PointXYZRGB p;

            p.x=pcloud[i].x;
             p.y=pcloud[i].y;
              p.z=pcloud[i].z;

              p.r = pcrgb[i][0];
               p.g = pcrgb[i][1];
                p.b = pcrgb[i][2];




              (*mycloud).push_back(p);
              //printf("Point: X:%.2f, Y:%.2f Z:%.2f\n",o.x,o.y,o.z);

        }




//reconstruct_surface(simplecloud);


       // printf("\nSIZE OF CLOUD: %ld", cld.size());

        viewer.showCloud (mycloud);

 while (!viewer.wasStopped ())
   {
   }


 for(int i=0;i<32555;i++)
  for(int j=0;j<32555;j++)
  ;
//	cout<<"total of " <<pcloud.size() <<" keypoints" << totpontos;



}
*/

/*
void load_pointcloud(vector<cloudPoint*> &Cloud)
{

  FILE * pFile;
  vector<Point3d> cloud;
  vector<Vec3b> rgb;


  vector<double> vx, vy,vz;
  //pFile = fopen ("img_mina/best_clouds/BA_supremo_166.txt","r+");
  pFile = fopen ("img_mina/point_cloud.txt","r+");
    double x=0,y=0,z=0;
    char r=0,g=0,b=0;



    while(!feof (pFile))
    {
      fscanf (pFile, "%lf %lf %lf %c %c %c ", &x,&y,&z,&r,&g,&b);
      fflush(stdin);
      cloud.push_back(Point3d(x,y,z));
      //vx.push_back(x);
      //vy.push_back(y);
      //vz.push_back(z);
      rgb.push_back(Vec3b(r,g,b));
    }

       fclose (pFile);

       //vector<cloudPoint*> Cloud;

       for(long i=0;i<cloud.size()-2;i++)
         {
             cloudPoint * cp = new cloudPoint;
             pKeypoint * pk = new pKeypoint;
             pk->R =  rgb[i][0];
             pk->G =  rgb[i][1];
             pk->B =  rgb[i][2];

             cp->is_in_cloud = true;
             cp->x = cloud[i].x;
             cp->y = cloud[i].y;
             cp->z = cloud[i].z;
             cp->owners.push_back(pk);

             Cloud.push_back(cp);
         }

         remove_outliersfromcloud(Cloud);
         //reconstruct_surface(Cloud); // //
         show_pointCloud(Cloud);

       //sort(vx.begin(),vx.end());
       // sort(vy.begin(),vy.end());
       // sort(vz.begin(),vz.end());

       // x = vx[(int)(vx.size()/2)];
       // y = vy[(int)(vy.size()/2)];
       // z = vz[(int)(vz.size()/2)];



    /*   pcl::visualization::CloudViewer viewer ("Sparse DEM");
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr mycloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for(long i=0;i<cloud.size();i++)
        {
          //  pcl::PointXYZRGB o;
             pcl::PointXYZRGB p;

            p.x=cloud[i].x; //-x; p.x*=1000;
             p.y=cloud[i].y; //-y; p.y*=1000;
              p.z=cloud[i].z;//-z; p.z*=1000;

              p.r = rgb[i][0];
               p.g = rgb[i][1];
                p.b = rgb[i][2];

              (*mycloud).push_back(p);

        }
    //printf("%.10f\n",cloud[0].x);
    viewer.showCloud (mycloud);

    while (!viewer.wasStopped ())
   {}
*/

//}

void save_pointcloud(vector<cloudPoint*> Cloud)
{
   FILE * pFile;

   pFile = fopen ("/home/itv/Sources/runs/run_small_mine/result/cloud.txt","w");


    for (long i=0; i<Cloud.size(); i++)
	{
      if(Cloud[i]->is_in_cloud) //&& Cloud[i]->x >0.31 && Cloud[i]->y >0.65 && Cloud[i]->z > 9.09 && Cloud[i]->x <0.32 && Cloud[i]->y <0.66 && Cloud[i]->z < 9.1  )
      {

        //fprintf (pFile, "%.10lf %.10lf %.10lf %c %c %c \n",(Cloud[i]->x-0.31)*100,(Cloud[i]->y-0.65)*100,(Cloud[i]->z-9.09)*100,Cloud[i]->owners[0]->R,Cloud[i]->owners[0]->G,Cloud[i]->owners[0]->B);
        fprintf (pFile, "%.10lf %.10lf %.10lf %c %c %c \n",Cloud[i]->x,Cloud[i]->y,Cloud[i]->z,Cloud[i]->owners[0]->R,Cloud[i]->owners[0]->G,Cloud[i]->owners[0]->B);
      }
   }
   fclose (pFile);


}

/*
void remove_outliersfromcloud(vector<cloudPoint*> Cloud)
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr mycloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

        for(long i=0;i<Cloud.size();i++)
        {

          if(Cloud[i]->is_in_cloud)
          {

             pcl::PointXYZ p;

             p.x=Cloud[i]->x;
             p.y=Cloud[i]->y;
             p.z=Cloud[i]->z;


            (*mycloud).push_back(p);
          }
        }

    // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (mycloud);
  sor.setMeanK (200); //200
  sor.setStddevMulThresh (4.0); //4
  sor.filter (*cloud_filtered);



  //std::cerr << "Cloud after filtering: " << std::endl;
  //std::cerr << *cloud_filtered << std::endl; //INLIERS
    for(long i=0,k=0;i<Cloud.size();i++)
    {
         if(Cloud[i]->is_in_cloud)
          {
              if( k < (*cloud_filtered).size() && (*cloud_filtered)[k].x == (float)Cloud[i]->x && (*cloud_filtered)[k].y == (float)Cloud[i]->y && (*cloud_filtered)[k].z == (float)Cloud[i]->z )
               k++;
              else
              Cloud[i]->is_in_cloud = false; //remove outlier

          }

    }

}
*/

void stereo_Rectify(Mat imgObj, Mat imgCena, Matx34d P0, Matx34d P1)
{

cv::Mat _R1, _P1, _R2, _P2, Q;
Size img_size = imgObj.size();
Rect roi1, roi2;
Mat dis = (Mat_<double>(1,5)<<0,0,0,0,0); //zero distortion



Mat_<double> R1(3,3),t1(3,1), R2(3,3), t2(3,1),R(3,3);
Mat t;

		R1(0,0) = P0(0,0); R1(0,1) = P0(0,1); R1(0,2) = P0(0,2); t1(0) = P0(0,3);
		R1(1,0) = P0(1,0); R1(1,1) = P0(1,1); R1(1,2) = P0(1,2); t1(1) = P0(1,3);
		R1(2,0) = P0(2,0); R1(2,1) = P0(2,1); R1(2,2) = P0(2,2); t1(2) = P0(2,3);

		R2(0,0) = P1(0,0); R2(0,1) = P1(0,1); R2(0,2) = P1(0,2); t2(0) = P1(0,3);
		R2(1,0) = P1(1,0); R2(1,1) = P1(1,1); R2(1,2) = P1(1,2); t2(1) = P1(1,3);
		R2(2,0) = P1(2,0); R2(2,1) = P1(2,1); R2(2,2) = P1(2,2); t2(2) = P1(2,3);

		Mat rvec1, rvec2,rvecf,tvecf;

        /*Eigen::Matrix3d mat3d1,mat3d2;
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                mat3d1(i,j)=R1(i,j);
                mat3d2(i,j)=R2(i,j);
            }
        }

        Eigen::AngleAxis<double> q1(mat3d1);
        Eigen::AngleAxis<double> q2(mat3d2);
        Eigen::AngleAxis<double> qr;


        Eigen::Matrix3d matfinal = qr.toRotationMatrix();

           for(int i=0;i<3;i++)
                for(int j=0;j<3;j++)
                  R(i,j)=matfinal(i,j);

          */
		//Rodrigues(R1,rvec1);
		//Rodrigues(R2,rvec2);
		//solve(R1,R2,R);
		//transpose(R2,R2);
		//R=R2*R1;

		//rvecf = rvec2-rvec1;
		//Rodrigues(rvecf,R);

		R= R1.inv() * R2;
		t = t2-t1;

stereoRectify( K, dis, K, dis, img_size, R, t, _R1, _R2, _P1, _P2, Q, CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2 );

Mat map11, map12, map21, map22;

initUndistortRectifyMap(K, dis, _R1, _P1, img_size, CV_16SC2, map11, map12);
initUndistortRectifyMap(K, dis, _R2, _P2, img_size, CV_16SC2, map21, map22);

Mat img1r, img2r;

remap(imgObj, img1r, map11, map12, INTER_LINEAR);
remap(imgCena, img2r, map21, map22, INTER_LINEAR);

string path = "img_mina/result/disp";
 char buffer2 [33];
    sprintf(buffer2,"%d",rectf_count++);
    path+=buffer2;
    path+=".jpg";

imwrite("img_mina/result/rectf1.jpg",img1r);
imwrite("img_mina/result/rectf2.jpg",img2r);




Mat disp;

 //sgbm(img1r,img2r,disp);

//imwrite("img1/disparity.jpg",disp);
/*
StereoSGBM sbm;
sbm.SADWindowSize = 5; //3 //15
sbm.numberOfDisparities = 1024;
sbm.preFilterCap = 63;
sbm.minDisparity = 0;
sbm.uniquenessRatio = 5;
sbm.speckleWindowSize = 200;
sbm.speckleRange = 32; //32
sbm.disp12MaxDiff = -1;
sbm.fullDP = false;
sbm.P1 = 216;
sbm.P2 = 864;

//img1r =imread("img_mina/result/view0.png");
//img2r =imread("img_mina/result/view2.png");

cvtColor( img1r, img1r, CV_BGR2GRAY );
cvtColor( img2r, img2r, CV_BGR2GRAY );

//img1r = img1r - img2r;
//imwrite("img1/diff.jpg",img1r);

//StereoBM sbm(sbm.BASIC_PRESET,80,21);



Mat disp8;
sbm(img1r,img2r, disp);
disp.convertTo(disp8, CV_8U, 255/((400)*16.));
//normalize(disp,disp8);

//disp8 = disp8*255;
//imshow("disp map",disp); waitKey(0);
imwrite(path,disp8);

/*
sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
sgbm->setMinDisparity(0);
sgbm->setNumDisparities(numberOfDisparities);
sgbm->setUniquenessRatio(10);
sgbm->setSpeckleWindowSize(100);
sgbm->setSpeckleRange(32);
sgbm->setDisp12MaxDiff(1);
sgbm->setMode(1);
*/
	// waitKey(0);

//correctMatches(fmat2, cam1, cam2, n_cam1, n_cam2);

//cv::triangulatePoints(P, P1, cam1, cam2, points3D);

}



/*
void reconstruct_surface(vector<cloudPoint*> Cloud)
{

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for(long i=0;i<Cloud.size();i++)
        {

          if(Cloud[i]->is_in_cloud)
          {

             pcl::PointXYZ p;

             p.x=Cloud[i]->x;
             p.y=Cloud[i]->y;
             p.z=Cloud[i]->z;


            (*cloud).push_back(p);
          }
        }



/*pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
mls.setInputCloud (cloud);
mls.setSearchRadius (0.01);
mls.setPolynomialFit (true);
mls.setPolynomialOrder (2);
mls.setUpsamplingMethod (pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
mls.setUpsamplingRadius (0.005);
mls.setUpsamplingStepSize (0.003);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed (new pcl::PointCloud<pcl::PointXYZ> ());
mls.process (*cloud_smoothed);
*/


/*pcl::PointCloud <pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PassThrough <pcl::PointXYZ> filter;
filter.setInputCloud(cloud);
filter.filter(*filtered);
*/
/*pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
ne.setNumberOfThreads (8);
ne.setInputCloud(cloud);//(cloud_smoothed);
ne.setRadiusSearch (0.5); //0.01
Eigen::Vector4f centroid;
compute3DCentroid (*cloud,centroid);//(*cloud_smoothed, centroid);

ne.setViewPoint (centroid[0], centroid[1], centroid[2]);

pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
ne.compute (*cloud_normals);

/*for( size_t i = 0; i < cloud_normals->size (); ++i)
{
cloud_normals->points[i].normal_x *= -1;
cloud_normals->points[i].normal_y *= -1;
cloud_normals->points[i].normal_z *= -1;
}*/

/*pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals (new pcl::PointCloud<pcl::PointNormal> ());

pcl::concatenateFields (*cloud/**cloud_smoothed*/ /*, *cloud_normals, *cloud_smoothed_normals);*/


/*pcl::Poisson<pcl::PointNormal> poisson;
poisson.setDepth (13); //9
poisson.setInputCloud(cloud_smoothed_normals);
pcl::PolygonMesh mesh;
poisson.reconstruct (mesh);
pcl::io::saveVTKFile ("img_mina/result/mesh.vtk", mesh);

}
*/


void load_reprjerr(vector<imgMatches*> cams, Mat cam_matrix, double img_height)
{

  for(int i=0;i<cams.size();i++)
        if(cams[i]->triangulated)
        {
            Mat_<double> KP1 = cam_matrix * Mat(cams[i]->P_mat); //Projection matrix


            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL && cams[i]->pkeypoints[j]->cloud_p->is_in_cloud)
              {

                 cloudPoint * cp = cams[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 Point2d cvp (cams[i]->pkeypoints[j]->pt.x,cams[i]->pkeypoints[j]->pt.y);

                cp->reprj_errs.push_back(abs(norm(cvp0-cvp)));

              }


        }

}

bool remove_outliers_carefully(vector<imgMatches*> cams, Mat cam_matrix, double img_height) //this function tries to keep good points even if they are bad in some cameras
{
   long qtd_rmv=0;
    long total = 0;
    bool go=false;


    for(int i=0;i<cams.size();i++)
        if(cams[i]->triangulated)
        {   total = 0;
            Mat_<double> KP1 = Kvec[i] * Mat(cams[i]->P_mat); //Projection matrix

             Mat pts(cams[i]->pkeypoints.size(),1,CV_32FC2);
            // pts.create(cams[i]->pkeypoints.size(),1,CV_32FC2);//

            for(int j=0;j<cams[i]->pkeypoints.size();j++) //points to undistort
            {

                 //Mat mt;
                 //mt.create(1,1,CV_32FC2);
                 pts.at<Vec2f>(j,0)[0] = cams[i]->pkeypoints[j]->pt.x;
                 pts.at<Vec2f>(j,0)[1] = cams[i]->pkeypoints[j]->pt.y;

				//if(j != pts.rows)
				//	cout<<"ESTOUROU FORTISSIMAMENTE"<<endl;

                // pts.push_back(mt);

            }


             cv::undistortPoints(pts,pts,Kvec[i],Dvec[i]);


            //cout<<"undistorting with K and D " <<endl<< Kvec[i]<<endl<<Dvec[i]<<endl;


            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL && cams[i]->pkeypoints[j]->cloud_p->is_in_cloud && cams[i]->pkeypoints[j]->master_idx == j)
              {
                 total++;
                 cloudPoint * cp = cams[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 Mat_<double> XK(3,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 XK(0) = pts.at<Vec2f>(j,0)[0]; XK(1) = pts.at<Vec2f>(j,0)[1]; XK(2)= 1.0;
                 Mat_<double> XKN = Kvec[i]*XK; // rectified pixel coords


                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 //Point2d cvp (cams[i]->pkeypoints[j]->pt.x,cams[i]->pkeypoints[j]->pt.y);
                 Point2d cvp (XKN(0),XKN(1));
                 //cout<<cams[i]->pkeypoints[j]->pt<<" is "<< XKN <<endl;

                 //cout<<cvp<<endl; getchar();


                 if(abs(norm(cvp0-cvp)) > confidence.first/(double)confidence.second || abs(norm(cvp0-cvp)) > UPPER_BOUND)//img_height * 0.0068  /*&& cp->can_ba == 1*/) //outlier 1.0    -- 0.00183823529 ->CAN_BA
                {

                 //cp->is_in_cloud = false;
                int k=0;

                    for(k =0; k<cp->owners.size() && cp->owners[k] != cams[i]->pkeypoints[j];k++);

                if(k < cp->owners.size())
                {
                    cp->owners.erase(cp->owners.begin() + k);

                    int l=0;

                    while(j+l < cams[i]->pkeypoints.size() && cams[i]->pkeypoints[j+l]->master_idx == cams[i]->pkeypoints[j]->master_idx)
                    {
                        cams[i]->pkeypoints[j+l]->cloud_p = NULL; //removing projection of that point
                        l++;
                    }
                }

                 qtd_rmv++;
                 total--;
                 }

                 if(cp->owners.size()<2)
                    cp->is_in_cloud = false;

              }

              if(total<15)
              {
                cams[i]->triangulated = false; // Remove this camera cuz it has high chances of being wrong
                cams[i]->already_rmv = true;
                go = true;
              }

        }

                cout<<"Filtered "<< qtd_rmv <<" bad projections from cameras."<<endl;
               // getchar();
                return go;

}



bool remove_outliers_by_reprojerror(vector<imgMatches*> cams, Mat cam_matrix, double img_height)
{
    long qtd_rmv=0;
    long qtd_recover = 0;
    long total = 0;
    bool go=false;

    if(!FILTER_POINTS) //deactivate outlier removal
    	return true;


    cout<<"-- Removing bad projections with threshold "<<(confidence.first/(double)confidence.second)<<" --"<<endl;

    for(int i=0;i<cams.size();i++)
        if(cams[i]->triangulated)
        {   total = 0;


            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL && cams[i]->pkeypoints[j]->cloud_p->is_in_cloud)// && cams[i]->pkeypoints[j]->master_idx == j)
              {
                 total++;
                 cloudPoint * cp = cams[i]->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = Mat(cams[i]->P_mat) * X;


				 float x_u = xPt_img(0)/xPt_img(2);
                 float y_u = xPt_img(1)/xPt_img(2); //perspective proj

                 
                double k1 = Dvec[i].at<double>(0,0);
   				double k2 = Dvec[i].at<double>(0,1);
   				double k3 = Dvec[i].at<double>(0,2);
    			double t1 = Dvec[i].at<double>(0,3);
   				double t2 = Dvec[i].at<double>(0,4);

                   // Apply distortion (xd,yd) = disto(x_u,y_u)
		        double r2 = x_u*x_u + y_u*y_u;
		        double r4 = r2 * r2;
		        double r6 = r4 * r2;
		        double r_coeff = (1.0 + k1*r2 + k2*r4 + k3*r6);
		        double t_x = t2 * (r2 + 2.0 * x_u*x_u) + 2.0 * t1 * x_u * y_u;
		        double t_y = t1 * (r2 + 2.0 * y_u*y_u) + 2.0 * t2 * x_u * y_u;;

		        double x_d = x_u * r_coeff + t_x;
		        double y_d = y_u * r_coeff + t_y;

			    float projected_x = Kvec[i].at<double>(0,2) + Kvec[i].at<double>(0,0) * x_d;
   			    float projected_y = Kvec[i].at<double>(1,2) + Kvec[i].at<double>(1,1) * y_d;


   			     Point2f estimated(projected_x,projected_y);
                 Point2f measurement = cams[i]->pkeypoints[j]->pt;

                 if(norm(estimated - measurement) > (confidence.first/(double)confidence.second) || norm(estimated - measurement) > UPPER_BOUND)//img_height * 0.0068  /*&& cp->can_ba == 1*/) //outlier 1.0    -- 0.00183823529 ->CAN_BA
                {

				 if(cams[i]->pkeypoints[j]->active)
					qtd_rmv++;

                	cams[i]->pkeypoints[j]->active = false;
                //cp->is_in_cloud = false;

                /*	cams[i]->pkeypoints[j]->cloud_p = NULL;

                	for(int m=0;m< cp->owners.size(); m++)
                	{
                		if(cp->owners[m] == cams[i]->pkeypoints[j])
                		{
                			cp->owners.erase(cp->owners.begin() + m);
                			break;
                		}
                	}

                	if(cp->owners.size() < 2)
                	 cp->is_in_cloud = false;
					*/



                 total--;
                 }

                 else if(!cams[i]->pkeypoints[j]->active)
                 {
                 	cams[i]->pkeypoints[j]->active = true;
                 	qtd_recover++;
                 }

              }


              if(total<15)
              {
                cams[i]->triangulated = false; // Remove this camera cuz it has high chances of being wrong
                cams[i]->already_rmv = true;
                go = true;
              }

        }

                cout<<"Filtered "<< qtd_rmv <<" bad 3D projs from cloud."<<endl;
                cout<<"Recovered "<<qtd_recover<<" Projs from cloud."<< endl;

                return go;

}

int test_imgfeatures(vector<imgMatches*> cams)
{
    int wrong =0 ;


    for(int i=0;i<cams.size();i++)
	 {
            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL)
              {

                 cloudPoint * cp = cams[i]->pkeypoints[j]->cloud_p;
                 if (cp->is_in_cloud)
                 wrong++;

                bool achou=false;

                 for(int k=0;k<cp->owners.size();k++)
                 {
                    if(cp->owners[k] == cams[i]->pkeypoints[j])
                        achou = true;
                 }

                 if(!achou)
                    wrong++;

               }

     }

     return wrong;
}


void correct_cloud(vector<cloudPoint*> &Cloud)
{
	for(size_t i=0;i<Cloud.size();i++)
	{
		if(Cloud[i]->is_in_cloud)
		{
			int good_projs = 0;
			for(int j=0;j<Cloud[i]->owners.size();j++)
			{
				if(Cloud[i]->owners[j]->img_owner->triangulated && Cloud[i]->owners[j]->active)
					good_projs++;

				if(good_projs>1)
				{

					break;
				}
			}

			if(good_projs<2)
				Cloud[i]->is_in_cloud = false;
		}
	}
}

bool remove_inconsistent_projs(vector<imgMatches*> cams)
{

    long total = 0;
    long all_projs=0;
    long qtd_rmv=0;
    bool go=false;


    for(int i=0;i<cams.size();i++)
	 {
            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL)
              {
              	 all_projs++;
                 qtd_rmv=0;
                 cloudPoint * cp = cams[i]->pkeypoints[j]->cloud_p;
                 int pos_cld;

                 for(int k=0;k<cp->owners.size();k++)
                 {
                 	pKeypoint* pk = cp->owners[k];

                 	if(pk->img_owner->id == cams[i]->pkeypoints[j]->img_owner->id && pk!=cams[i]->pkeypoints[j] ) //inconsistency
                 	{
                 		pk->cloud_p = NULL;
                 		cp->owners.erase(cp->owners.begin()+k); k--;
                 		qtd_rmv++;
                 	}

                 }

                 	total+=qtd_rmv;

              }


         }


                cout<<"Filtered "<< total <<" bad projs from cameras in a total of "<<all_projs<<endl; //getchar();

}


double remove_outliers_from_cam(imgMatches* cam, Mat cam_matrix, Mat dist_coeff, double max_err)
{
	double qtd_rmv=0;
    double qtd_tot=0;

    if(!FILTER_POINTS)
    	return 0;


     double k1 = dist_coeff.at<double>(0,0);
     double k2 = dist_coeff.at<double>(0,1);
     double k3 = dist_coeff.at<double>(0,2);
     double t1 = dist_coeff.at<double>(0,3);
     double t2 = dist_coeff.at<double>(0,4);

     cout<<"Dist coefficients: "<<k1<<" "<<k2<<" "<<k3<< " "<<t1<<" "<<t2<<endl;

	for(int j=0;j<cam->pkeypoints.size();j++)
              if(cam->pkeypoints[j]->cloud_p!=NULL && cam->pkeypoints[j]->cloud_p->is_in_cloud)
              {
              //   total++;
                 qtd_tot++;
                 cloudPoint * cp = cam->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = Mat(cam->P_mat) * X;


                 float x_u = xPt_img(0)/xPt_img(2);
                 float y_u = xPt_img(1)/xPt_img(2); //perspective proj


                 // Apply distortion (xd,yd) = disto(x_u,y_u)
		        double r2 = x_u*x_u + y_u*y_u;
		        double r4 = r2 * r2;
		        double r6 = r4 * r2;
		        double r_coeff = (1.0 + k1*r2 + k2*r4 + k3*r6);
		        double t_x = t2 * (r2 + 2.0 * x_u*x_u) + 2.0 * t1 * x_u * y_u;
		        double t_y = t1 * (r2 + 2.0 * y_u*y_u) + 2.0 * t2 * x_u * y_u;;

		        double x_d = x_u * r_coeff + t_x;
		        double y_d = y_u * r_coeff + t_y;

			    float projected_x = cam_matrix.at<double>(0,2) + cam_matrix.at<double>(0,0) * x_d;
   			    float projected_y = cam_matrix.at<double>(1,2) + cam_matrix.at<double>(1,1) * y_d;


   			     Point2f estimated(projected_x,projected_y);
                 Point2f measurement = cam->pkeypoints[j]->pt;

                 double m_error = norm(estimated - measurement);

                 if(m_error > max_err) //outlier 1.0    -- 0.00183823529 ->CAN_BA
                 {
                 	qtd_rmv++;
                 	//cp->is_in_cloud = false;
                 }


              }


              cout<<""<< qtd_rmv <<" large error points from this camera in a total of "<<qtd_tot<<" projections."<<endl;


              if((double)qtd_rmv/(double)qtd_tot < 0.75) //remove them
              {
              	qtd_rmv =0;
              	qtd_tot = 0;

	              for(int j=0;j<cam->pkeypoints.size();j++)
	              if(cam->pkeypoints[j]->cloud_p!=NULL && cam->pkeypoints[j]->cloud_p->is_in_cloud)
	              {
	              //   total++;
	                 qtd_tot++;
	                 cloudPoint * cp = cam->pkeypoints[j]->cloud_p;
	                 Mat_<double> X(4,1);
	                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
	                 Mat_<double> xPt_img = Mat(cam->P_mat) * X;



	                 float x_u = xPt_img(0)/xPt_img(2);
	                 float y_u = xPt_img(1)/xPt_img(2); //perspective proj

              	   // Apply distortion (xd,yd) = disto(x_u,y_u)
		   		     double r2 = x_u*x_u + y_u*y_u;
		   		     double r4 = r2 * r2;
		   		     double r6 = r4 * r2;
		   		     double r_coeff = (1.0 + k1*r2 + k2*r4 + k3*r6);
		    	     double t_x = t2 * (r2 + 2.0 * x_u*x_u) + 2.0 * t1 * x_u * y_u;
		        	 double t_y = t1 * (r2 + 2.0 * y_u*y_u) + 2.0 * t2 * x_u * y_u;;

		       		 double x_d = x_u * r_coeff + t_x;
		      		 double y_d = y_u * r_coeff + t_y;

			    	 float projected_x = cam_matrix.at<double>(0,2) + cam_matrix.at<double>(0,0) * x_d;
   			       	 float projected_y = cam_matrix.at<double>(1,2) + cam_matrix.at<double>(1,1) * y_d;


	   			     Point2f estimated(projected_x,projected_y);
	                 Point2f measurement = cam->pkeypoints[j]->pt;

	                 if(norm(estimated - measurement) > max_err) //outlier 1.0    -- 0.00183823529 ->CAN_BA
	                 {
	                	qtd_rmv++;
	                	/*
	                	int k=0;

	                    for(k =0; k<cp->owners.size() && cp->owners[k] != cam->pkeypoints[j];k++);

	                    if(k < cp->owners.size())
	                    {
	                        cp->owners.erase(cp->owners.begin() + k);
							cam->pkeypoints[j]->cloud_p = NULL; //removing projection of that point
	                    }

	                    if(cp->owners.size()<2)
	                    cp->is_in_cloud = false;
	                    */
	                	cam->pkeypoints[j]->active = false;


	                 }

	            }

          }

     cout<<""<< qtd_rmv <<" removed projections from this camera in a total of "<<qtd_tot<<"."<<endl;

     return (double)qtd_rmv/(double)qtd_tot;
}

/*double remove_outliers_from_cam(imgMatches* cam, Mat cam_matrix, Mat dist_coeff, double max_err)
{



            Mat_<double> KP1 = cam_matrix * Mat(cam->P_mat); //Projection matrix

             Mat pts;

             double qtd_rmv=0;
             double qtd_tot=0;


            for(int j=0;j<cam->pkeypoints.size();j++) //points to undistort
            {

                 Mat mt;
                 mt.create(1,1,CV_32FC2);
                 mt.at<Vec2f>(0,0)[0] = cam->pkeypoints[j]->pt.x;
                 mt.at<Vec2f>(0,0)[1] = cam->pkeypoints[j]->pt.y;

                 pts.push_back(mt);
            }


             cv::undistortPoints(pts,pts,cam_matrix,dist_coeff);


           //  cout<<"undistorting with coeffs " <<Dvec[i]<<endl;


             //computing outlier ratio

              for(int j=0;j<cam->pkeypoints.size();j++)
              if(cam->pkeypoints[j]->cloud_p!=NULL && cam->pkeypoints[j]->cloud_p->is_in_cloud)
              {
              //   total++;
                 qtd_tot++;
                 cloudPoint * cp = cam->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 Mat_<double> XK(3,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 XK(0) = pts.at<Vec2f>(j,0)[0]; XK(1) = pts.at<Vec2f>(j,0)[1]; XK(2)= 1.0;
                 Mat_<double> XKN = cam_matrix*XK; // rectified pixel coords


                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 //Point2d cvp (cams[i]->pkeypoints[j]->pt.x,cams[i]->pkeypoints[j]->pt.y);
                 Point2d cvp (XKN(0),XKN(1));
                 //cout<<cams[i]->pkeypoints[j]->pt<<" is "<< XKN <<endl;

                 //cout<<cvp<<endl; getchar();


                 if(abs(norm(cvp0-cvp)) > max_err) //outlier 1.0    -- 0.00183823529 ->CAN_BA
                	qtd_rmv++;

              }




           if(qtd_rmv/qtd_tot < 0.75)
           { qtd_tot =0; qtd_rmv=0;
            for(int j=0;j<cam->pkeypoints.size();j++)
              if(cam->pkeypoints[j]->cloud_p!=NULL && cam->pkeypoints[j]->cloud_p->is_in_cloud)
              {
              //   total++;
                 qtd_tot++;
                 cloudPoint * cp = cam->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 Mat_<double> XK(3,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 XK(0) = pts.at<Vec2f>(j,0)[0]; XK(1) = pts.at<Vec2f>(j,0)[1]; XK(2)= 1.0;
                 Mat_<double> XKN = cam_matrix*XK; // rectified pixel coords


                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 //Point2d cvp (cams[i]->pkeypoints[j]->pt.x,cams[i]->pkeypoints[j]->pt.y);
                 Point2d cvp (XKN(0),XKN(1));
                 //cout<<cams[i]->pkeypoints[j]->pt<<" is "<< XKN <<endl;

                 //cout<<cvp<<endl; getchar();


                 if(abs(norm(cvp0-cvp)) > max_err) //outlier 1.0    -- 0.00183823529 ->CAN_BA
                {
                    //cp->is_in_cloud = false;

                     int k=0;

                    for(k =0; k<cp->owners.size() && cp->owners[k] != cam->pkeypoints[j];k++);

                    if(k < cp->owners.size())
                    {
                        cp->owners.erase(cp->owners.begin() + k);
						cam->pkeypoints[j]->cloud_p = NULL; //removing projection of that point
                    }

                    if(cp->owners.size()<2)
                    cp->is_in_cloud = false;


                     qtd_rmv++;


                 }

              }

          }


                cout<<""<< qtd_rmv <<" bad points from this camera in a total of "<<qtd_tot<<" projections."<<endl;

                return (double)qtd_rmv/(double)qtd_tot;

}
*/






void remove_outliers_by_reprojerror_2(vector<imgMatches*> cams, Mat cam_matrix, double img_height,vector<cloudPoint*>Cloud)
{
    long qtd_rmv = 0;

	for(long i=0;i<Cloud.size();i++)
	{
  	 if(Cloud[i]->is_in_cloud)
          for(int j=0;j<Cloud[i]->owners.size();j++)
	  {
	    if(Cloud[i]->owners[j]->img_owner->triangulated)
	   {

	    Mat_<double> KP1 = Kvec[Cloud[i]->owners[j]->img_owner->id] * Mat(Cloud[i]->owners[j]->img_owner->P_mat); //Projection matrix


                 cloudPoint * cp = Cloud[i];
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;                                //reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 Point2d cvp (Cloud[i]->owners[j]->pt.x,Cloud[i]->owners[j]->pt.y);

                 if(abs(norm(cvp0-cvp)) > img_height * 0.00183823529) //outlier 1.0
                 {
                    cp->is_in_cloud = false;
                    qtd_rmv++;
                    }
	       }
             }


        }

        cout<<"Filtered "<< qtd_rmv <<" bad points from cloud."<<endl;
}


double camera_reprojerr(imgMatches* cam, Mat cam_matrix, double img_height)
{
	vector<double>reprj_errs;

            Mat_<double> KP1 = cam_matrix * Mat(cam->P_mat); //Projection matrix


            for(int j=0;j<cam->pkeypoints.size();j++)
              if(cam->pkeypoints[j]->cloud_p!=NULL && cam->pkeypoints[j]->cloud_p->is_in_cloud)
              {
                 cloudPoint * cp = cam->pkeypoints[j]->cloud_p;
                 Mat_<double> X(4,1);
                 X(0) = cp->x; X(1) = cp->y; X(2) = cp->z; X(3) = 1.0;
                 Mat_<double> xPt_img = KP1 * X;				//reproject
                 Point2d cvp0(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

                 Point2d cvp (cam->pkeypoints[j]->pt.x,cam->pkeypoints[j]->pt.y);

                 reprj_errs.push_back(abs(norm(cvp0-cvp)));



              }


	std::sort(reprj_errs.begin(),reprj_errs.end());

    if(reprj_errs.size()>0)
	return reprj_errs[reprj_errs.size()/2];

	else
	return 0;
}


void SFM2PMVS (vector<imgMatches*> imgs_features, Mat k, string img_dir, vector<string> imgs_name, vector<Mat> imgs, char * output_path, double downscale_factor)
{
    //char * output_path = "/home/itv/Sources/runs/run_small_mine/result";
     //mkdir(output_path, 0770);
     //char buf[1024];
     //sprintf(buf, "%s/prep_pmvs.sh", output_path);
     //FILE *f_scr = fopen(buf, "w");
     //fprintf(f_scr, "# Script for preparing images and calibration data \n # for Yasutaka Furukawa's PMVS system\n\n");
     /*fprintf(f_scr, "BUNDLER_BIN_PATH="" # Edit this line before running\n");
     fprintf(f_scr, "if [ \"$BUNDLER_BIN_PATH\" == \"\" ] ; then "
    "echo Please edit prep_pmvs.sh to specify the path to the "
    " bundler binaries.; exit; fi\n");*/
     //fprintf(f_scr, "# Apply radial undistortion to the images\n");
     //fprintf(f_scr, "$BUNDLER_BIN_PATH/RadialUndistort %s %s %s\n", list_file, bundle_file, output_path);
     //fprintf(f_scr, "\n# Create directory structure\n");
     // fprintf(f_scr, "mkdir -p %s/\n", output_path);
     //fprintf(f_scr, "mkdir -p %s/txt/\n", output_path);
     //fprintf(f_scr, "mkdir -p %s/visualize/\n", output_path);
     //fprintf(f_scr, "mkdir -p %s/models/\n", output_path);
     //fprintf(f_scr, "\n# Copy and rename files\n");
     int count = 0;
	char buf2[1024];
	sprintf(buf2,"%s", output_path);
//	mkdir(buf2,0770);
	sprintf(buf2,"%s/txt",output_path);
//	mkdir(buf2,0770);
	sprintf(buf2,"%s/visualize",output_path);
   //     mkdir(buf2,0770);
        sprintf(buf2,"%s/models",output_path);
//	mkdir(buf2,0770);

	//fclose(f_scr);
	sprintf(buf2,"%s/list.txt",output_path);
	FILE *f_imgs = fopen(buf2,"w");


     for (int i=0;i<imgs_features.size();i++)
     {
	if(imgs_features[i]->triangulated)
       {
        char buf[256];
        sprintf(buf, "%s/txt/%08d.txt", output_path, count);
       // char buf2[256];
       // sprintf(buf2, "%s/%s", img_dir.c_str(), imgs_name[i].c_str());
        FILE *f = fopen(buf, "w");
        assert(f);
	assert(f_imgs);
        Mat_<double> KP = Kvec[i] * Mat(imgs_features[i]->P_mat);
        fprintf(f, "CONTOUR\n");
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n", KP(0), KP(1), KP(2), KP(3));
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n", KP(4), KP(5), KP(6), KP(7));
        fprintf(f, "%0.6f %0.6f %0.6f %0.6f\n\n", KP(8), KP(9), KP(10), KP(11));

        char buffer [256];
        sprintf(buffer, "%s/visualize/%08d.jpg", output_path, count);

        //undistortion
        cv::Mat m_,undistorted;
        m_ = open_cv_image(img_dir+"/result/undistorted/"+imgs_name[i], 1.0);//downscale_factor);

        undistort(m_,undistorted,Kvec[i],Dvec[i]);

        imwrite(buffer, undistorted);//imgs[i]);//imgs[i]);
		fprintf(f_imgs,"%s\n",buffer);
       // fprintf(f_scr, "mv %s %s/txt/\n", buf, output_path);

        count++;
	fclose(f);
	}
     }

	//fclose(f_scr);
	fclose(f_imgs);

	char buf[256];
     /* Write the options file */
     sprintf(buf, "%s/pmvs_options.txt", output_path);
     FILE *f_opt = fopen(buf, "w");
     fprintf(f_opt, "level 1\n");
     fprintf(f_opt, "csize 2\n");
     fprintf(f_opt, "threshold 0.7\n");
     fprintf(f_opt, "wsize 7\n");
     fprintf(f_opt, "minImageNum 3\n");
     fprintf(f_opt, "CPU 8\n");
     fprintf(f_opt, "setEdge 0\n");
     fprintf(f_opt, "useBound 0\n");
     fprintf(f_opt, "useVisData 0\n");
     fprintf(f_opt, "sequence -1\n");
     fprintf(f_opt, "timages -1 0 %d\n", count);
     fprintf(f_opt, "oimages -3\n");
     fclose(f_opt);
}


void SFM2BUNDLER (vector<imgMatches*> imgs_features, Mat k, string img_dir, vector<string> imgs_name, vector<Mat> imgs, vector<cloudPoint*> Cloud, char * output_path )
{
//    char *output_path = "/home/itv/Sources/runs/run_small_mine/result";
        //mkdir(output_path, 0770);
	int c_cams=0;
	long c_pts =0;

	for (int i=0;i<imgs_features.size();i++)
    	if(imgs_features[i]->triangulated)
    		c_cams++;

    for(long i=0;i<Cloud.size();i++)
    	if(Cloud[i]->is_in_cloud)
    		c_pts++;

    char bufr[256];
    sprintf(bufr, "%s/bundle.out", output_path);
    FILE *fl = fopen(bufr, "w");
    fprintf(fl, "# Bundle file v0.3\n");
    fprintf(fl, "%d %ld\n",c_cams,c_pts);
    double k1=0;
    double k2=0;
    int pos=0;

    for (int i=0;i<imgs_features.size();i++)
    {
    	if(imgs_features[i]->triangulated)
    	{
        assert(fl);
        Mat_<double> P = Mat(imgs_features[i]->P_mat);
        fprintf(fl, "%0.6f %0.6f %0.6f\n",Kvec[i].at<double>(0,0),Dvec[i].at<double>(0,0),k2);
        fprintf(fl, "%0.6f %0.6f %0.6f\n", P(0), P(1), P(2));
        fprintf(fl, "%0.6f %0.6f %0.6f\n", -P(4), -P(5), -P(6));
        fprintf(fl, "%0.6f %0.6f %0.6f\n", -P(8), -P(9), -P(10));
        fprintf(fl, "%0.6f %0.6f %0.6f\n", P(3), -P(7), -P(11));
        imgs_features[i]->id = pos++;
    	}
    }

        for (int i=0;i<Cloud.size();i++) {
	if(Cloud[i]->is_in_cloud)
	{
        fprintf(fl, "%0.6f %0.6f %0.6f\n", Cloud[i]->x, Cloud[i]->y, Cloud[i]->z);
        fprintf(fl, "%d %d %d\n", Cloud[i]->owners[0]->R, Cloud[i]->owners[0]->G, Cloud[i]->owners[0]->B);
        fprintf(fl, "%d ", Cloud[i]->owners.size());
        for (int j=0;j<Cloud[i]->owners.size();j++)
        {
        	int img_idx = Cloud[i]->owners[j]->img_owner->id;
            fprintf(fl, "%d %d %0.4f %0.4f ", Cloud[i]->owners[j]->img_owner->id, j, Cloud[i]->owners[j]->pt.x - (imgs_features[img_idx]->cols/2.0), -Cloud[i]->owners[j]->pt.y + (imgs_features[img_idx]->rows/2.0) );
        }
	}
    }
    	char bufr2 [512];

    	sprintf(bufr2,"~/Sources/bundler_sfm/bin/./Bundle2Ply %s ~/Dropbox/Projetos/PLYs/bundle_out.ply",bufr);

    	//system(bufr2);
}


void save_elapsed_time(char * output_path)
{

	char bufr[256];
         sprintf(bufr, "%s/stats.txt", output_path);
         FILE *stats = fopen(bufr,"w");

 	//gettimeofday(&end, NULL);
     //   seconds  = end.tv_sec  - start.tv_sec;
    //	useconds = end.tv_usec - start.tv_usec;

    //	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

        fprintf(stats, "Total %d milliseconds / %0.4f hours",mtime,(mtime/1000.0)/3600.0);
        fclose(stats);

}

void remove_old_keypoints(Mat G, vector<imgMatches*> imgs_features, double percentile)
{
   for(int k = 0; k< imgs_features.size();k++)
   {

    if(!imgs_features[k]->already_rmv)
    {



    bool rem = true;
    for(int i=0;i<G.rows;i++) //triangulating image to connected vertices
        {
           if( (G.at<double>(i,k)>0 || G.at<double>(k,i)>0 ) && !imgs_features[i]->triangulated)
           {
               rem = false;
           }

        }

        if(rem) //remove
        {

              for(long j=0;j<imgs_features[k]->pkeypoints.size();j++)
                {
                    if(imgs_features[k]->pkeypoints[j]->cloud_p!=NULL && imgs_features[k]->pkeypoints[j]->cloud_p->is_in_cloud)
                        {
                            if(rand()%100 < 40) //remove it
                                imgs_features[k]->pkeypoints[j]->cloud_p->is_in_cloud = false;

                        }

                }

                imgs_features[k]->already_rmv =  true;
        }
   }

   }
}

double ms_error(vector<double> reproj_error)
{
    double tot=0;

    for(long i=0;i<reproj_error.size();i++)
    {
        tot+= reproj_error[i]*reproj_error[i];
    }

    tot/=(double)reproj_error.size();
}

bool sort_cld_by_projs(cloudPoint *c1, cloudPoint *c2)
{
    return (c1->owners.size() < c2->owners.size());
}

 //reduces the 3D points until there is at least k projs for each camera,
//trying to keep the 3D points that have the highest number of projections
void k_reduction(vector<cloudPoint*> Cloud, vector<imgMatches*> cams)
{
    std::sort(Cloud.begin(), Cloud.end(), sort_cld_by_projs);
    int th_min = 250;
    long c_rmv = 0;


        for(int i=0;i<cams.size();i++)
        {
            cams[i]->qtd_projs=0;

            for(int j=0;j<cams[i]->pkeypoints.size();j++)
              if(cams[i]->pkeypoints[j]->cloud_p!=NULL && cams[i]->pkeypoints[j]->cloud_p->is_in_cloud)
              {
                cams[i]->qtd_projs++;
              }

        }



    for(long i=0;i<Cloud.size();i++)
    {
        bool rmv = true;
            if(Cloud[i]->is_in_cloud)
            {
                for(int j=0;j<Cloud[i]->owners.size();j++)
                {
                    if(Cloud[i]->owners[j]->img_owner->qtd_projs - 1 < th_min)
                    {rmv = false; break;}
                }

                if(rmv)
                {
                    for(int j=0;j<Cloud[i]->owners.size();j++)
                    {
                        Cloud[i]->owners[j]->img_owner->qtd_projs--;

                    }

                    Cloud[i]->is_in_cloud = false;
                    c_rmv++;

                }
            }

    }

    cout<<"K reduction filtered "<<c_rmv<<" points."<<endl;
}

bool sort_good_pair(pair<double,pair<int,int> > e1, pair<double,pair<int,int> > e2)
{
    return e1.first>e2.first;
}

bool sort_good_pares(pair<double, pair<double,pair<int,int> > > e1, pair<double, pair<double,pair<int,int> > > e2)
{
    return e1.first>e2.first;
}

vector<Point2f> normalize_points(vector<Point2f> cam, Mat Kmat)
{
	vector<Point2f> out;


	for(int i=0;i<cam.size();i++)
	{

		Point2f novo;

		novo.x = cam[i].x - Kmat.at<double>(0,2);
		novo.y = cam[i].y - Kmat.at<double>(1,2);

		novo.x/=Kmat.at<double>(0,0);
		novo.y/=Kmat.at<double>(1,1);

		out.push_back(novo);
	}

	return out;
}

void find_good_baseline_pair(cv::Mat G, vector<imgMatches*> imgs_features, vector<vector<pairMatch*> > pairs ,std::pair<int,int> &highest_inlier)
{

vector<pair<double,pair<int,int> > > pares;
vector<pair<double, pair<double,pair<int,int> > > > good_pares;
vector<double> reproj_error;

double current_reprojerr;
double lowest_reprojerr=9999;
vector<pair<int,int> > nice_pairs;
pair<int,int> best_pair;
int c_inliers=-1;

    for(int i=0;i<G.rows;i++)
     for(int j=i+1;j<G.cols;j++)
     {
        if(G.at<double>(i,j)>0)
        pares.push_back(make_pair(G.at<double>(i,j),make_pair(i,j)));
     }


     std::sort(pares.begin(),pares.end(),sort_good_pair);

     for(int k=0;k<pares.size()*0.33;k++)
     {
     	int i,j;
     	i = pares[k].second.first;
     	j = pares[k].second.second;


     	 if(imgs_features[i]->EXIF && imgs_features[j]->EXIF || imgs_features[i]->had_intrinsics && imgs_features[j]->had_intrinsics)
     		good_pares.push_back(make_pair(10000.0 + G.at<double>(i,j)/(float)(pairs[i][j]->homog_inliers),pares[k]));
     	 else //Pairs without intrinsics considering in the last case in the sorting even if the ratio is good
     	 	good_pares.push_back(make_pair(G.at<double>(i,j)/(float)(pairs[i][j]->homog_inliers),pares[k]));
     }

     std::sort(good_pares.begin(),good_pares.end(),sort_good_pares);

     for(int m=0;m < 60 ;m++) // get m samples from the best 'k' goood pairs
     {

     int k = rand()%(int)(good_pares.size()*0.25);

     int i,j;

     vector<bool> inliers;

     i = good_pares[k].second.second.first;
     j = good_pares[k].second.second.second;

    // if(imgs_features[i]->EXIF && imgs_features[j]->EXIF || imgs_features[i]->had_intrinsics && imgs_features[j]->had_intrinsics)
     //{//
     	 vector<uchar> vec_inliers(pairs[i][j]->cam1.size(),0);
     	 int c_in=0;
    	 pairs[i][j]->F_mat = cv::findEssentialMat(normalize_points(pairs[i][j]->cam1,Kvec[i]), normalize_points(pairs[i][j]->cam2,Kvec[j]), 1.0, Point2d(0,0), cv::RANSAC, 0.999,(0.006*(double)imgs_features[i]->cols)/Kvec[i].at<double>(0,0),vec_inliers);

    	 for(int t=0; t<vec_inliers.size();t++)
    	 	if(vec_inliers[t])
    	 		c_in++;

    	 	//cout<<"Found "<<c_in<<" inliers for Essential mat. and " << pairs[i][j]->cam1.size()<< " for fundamental."<<endl;

          imgs_features[i]->P_mat = Matx34d (1,0,0,0,
                                             0,1,0,0,
                                             0,0,1,0);

         if(!find_P1(imgs_features[j]->P_mat, pairs[i][j]->F_mat,pairs[i][j]->cam1, pairs[i][j]->cam2,reproj_error,i,j,inliers))
         	continue;

         current_reprojerr = reproj_error[(int)(reproj_error.size()*0.75)]; //reproj_error[reproj_error.size()/2];//mean(reproj_error)[0];

         int n_of_inliers = 0;

         for(int t=0;t<inliers.size();t++)
         	if(inliers[t])
         		n_of_inliers++;

         if(n_of_inliers > c_inliers)
         {
         	c_inliers = n_of_inliers;
         	best_pair = make_pair(i,j);
         }

         // nice_pairs.push_back(make_pair(i,j));
     }

     highest_inlier = best_pair;//nice_pairs[rand()%nice_pairs.size()]; //nao get a random init good pair


     if(!(imgs_features[highest_inlier.first]->had_intrinsics && imgs_features[highest_inlier.second]->had_intrinsics))
     {
     	cout<<"WARNING: SEED PAIR DOESNT HAVE INTRINSICS"<<endl;
     }
    // getchar();
}


void remove_projs_lessthan(vector<cloudPoint*> &Cloud, int lessthan) //O(n) function
{

    vector<cloudPoint*> new_cloud;

     for(long i=0;i<Cloud.size();i++)
    {
        if(Cloud[i]->owners.size() >= lessthan)
        {
            new_cloud.push_back(Cloud[i]);
        }
        else //delete it and dont add it to new cloud
        {
            for(int j=0;j<Cloud[i]->owners.size();j++)
           {
               Cloud[i]->owners[j]->cloud_p = NULL;
           }

          Cloud[i]->owners.clear();
          delete(Cloud[i]);

        }
    }


    Cloud = new_cloud; // replace cloud with new cloud O(n)
}



cv::Mat find_ray(Matx34d P1,  Point2f n_x1) //Normalized P_mat and normalized point
{


cv::Mat_<double> R(3,3);
cv::Mat_<double> t(3,1);
cv::Mat_<double> px_pos(3,1);
cv::Mat_<double> wd_pos(3,1);
cv::Mat_<double> c(3,1);


	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			R(i,j) = P1(i,j);

t(0,0) = P1(0,3); t(1,0) = P1(1,3); t(2,0) = P1(2,3);

px_pos(0,0) = n_x1.x; px_pos(1,0) = n_x1.y; px_pos(2,0) = 1.0; //position of the projection in camera coord system, considering unit focal length z=1.0


c = -R.inv()*t; // camera center in world's coordinate


wd_pos = (R.inv()*px_pos) + c; // projection location in world's coordinate

cv::Mat_<double> ray = c - wd_pos;

return ray;
}

cv::Mat find_ray_3d(Matx34d P1,  Mat _X) //Normalized P_mat and 3D point position
{


cv::Mat_<double> R(3,3);
cv::Mat_<double> t(3,1);
cv::Mat_<double> c(3,1);
cv::Mat_<double> X(3,1);

X(0) = _X.at<double>(0,0);
X(1) = _X.at<double>(1,0);
X(2) = _X.at<double>(2,0);


	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			R(i,j) = P1(i,j);

t(0,0) = P1(0,3); t(1,0) = P1(1,3); t(2,0) = P1(2,3);


c = -R.inv()*t; // camera center in world's coordinate

cv::Mat_<double> ray = c - X;

return ray;
}

double angle_between_vec(Mat v1, Mat v2)
{
    //v1.at<double>(0,0) = 0; v1.at<double>(1,0) = 0; v1.at<double>(2,0) = 1;
    //v2.at<double>(0,0) = 0; v2.at<double>(1,0) = 1; v2.at<double>(2,0) = 0;


	double dotp = v1.at<double>(0,0)*v2.at<double>(0,0) + v1.at<double>(1,0)*v2.at<double>(1,0) + v1.at<double>(2,0)*v2.at<double>(2,0);
	double vl1 = norm(v1);
	double vl2 = norm(v2);

	double cos_th = dotp/(vl1*vl2);

	double theta = acos(cos_th)*(180.0/M_PI);

    //cout<<"Angle "<<theta<<endl;
	//getchar();

	return theta;


}

cv::Mat get_mean_rotation(vector<imgMatches*> imgs) //get mean rotation between cameras using SVD
{

    cv::Mat sum_R = Mat::zeros(3,3,CV_64F);
    double ct = 0;

    for(int i=0;i<imgs.size();i++)
    {
        if(imgs[i]->triangulated)
        {
        for(int j=0;j<3;j++)
            for(int k=0;k<3;k++)
                sum_R.at<double>(j,k)+=imgs[i]->P_mat(j,k);
        }
    }


    cv::SVD svd(sum_R,cv::SVD::MODIFY_A);

    cv::Mat avg_R = svd.u*svd.vt;

    return avg_R;
}

void export_points_wGPS(vector<cloudPoint*> Cloud, cv::Mat R, vector<Point2d> gps_data)
{
    FILE *f = fopen("/home/vnc/Dropbox/Projetos/PLYs/points_gps_out.grid", "w");

    Mat_<double> X(3,1);

    long total_pts=0;


        for(int i=0;i<Cloud.size();i++)
            if(Cloud[i]->is_in_cloud)
                total_pts++;


     fprintf(f,"%ld\n",total_pts);


    for(int i=0;i<Cloud.size();i++)
    {
        if(Cloud[i]->is_in_cloud)
        {
            X(0,0) = Cloud[i]->x;
            X(1,0) = Cloud[i]->y;
            X(2,0) = Cloud[i]->z;

            cv::Mat Xn = R*X;


            fprintf(f,"%f %f %f %d",Xn.at<double>(0,0),Xn.at<double>(1,0),Xn.at<double>(2,0),Cloud[i]->owners.size());


            for(int j=0;j<Cloud[i]->owners.size();j++)
            {
                fprintf(f," %f %f",gps_data[Cloud[i]->owners[j]->img_owner->id].x,gps_data[Cloud[i]->owners[j]->img_owner->id].y);
            }

            fprintf(f,"\n");

        }
    }

    fclose(f);

}

void save_GPS_list(vector<Point3d> gps_list, string output)
{
    FILE *f = fopen( (output + "gps_list.txt").c_str(), "w");

    for(unsigned int i=0;i<gps_list.size();i++)
    {
        fprintf(f,"%.12f %.12f %.12f\n",gps_list[i].x, gps_list[i].y,gps_list[i].z);
    }

    fclose(f);
}

void save_PLY( vector<imgMatches*> imgs_features, vector<cloudPoint*> Cloud, vector<Point2d> gps_data, string out_path)
{


bool save_camera_planes = false;

static char ply_header[] =
"ply\n"
"format ascii 1.0\n"
"element face 0\n"
"property list uchar int vertex_indices\n"
"element vertex %ld\n"
"property float x\n"
"property float y\n"
"property float z\n"
"property uchar diffuse_red\n"
"property uchar diffuse_green\n"
"property uchar diffuse_blue\n"
"end_header\n";
long num_points_out = 0;
string content;
std::stringstream ss;

cv::Mat_<double> R(3,3);
cv::Mat_<double> t(3,1);
cv::Mat_<double> c(3,1);

cv::Mat_<double> px_pos(3,1);
cv::Mat_<double> wd_pos(3,1);

FILE *fcamera = fopen( (out_path + "camera_pos_list.txt").c_str(), "w");
FILE *fpoints = fopen( (out_path + "points_pos_list.txt").c_str(), "w");

for(int i=0;i<imgs_features.size();i++)
{
	if(imgs_features[i]->triangulated)
	{
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				R(j,k) = imgs_features[i]->P_mat(j,k);

		t(0,0) = imgs_features[i]->P_mat(0,3); t(1,0) = imgs_features[i]->P_mat(1,3); t(2,0) = imgs_features[i]->P_mat(2,3);

		c = -R.inv()*t; // camera center in world's coordinate

		ss<<c.at<double>(0,0)<<" "<<c.at<double>(1,0)<<" "<<c.at<double>(2,0)<<" 255 0 255"<<endl;
		fprintf(fcamera,"%.12f %.12f %.12f\n",c.at<double>(0,0),c.at<double>(1,0),c.at<double>(2,0));

		num_points_out++;

		/////////////////////// camera projections in camera plane f = 1.0////////////////

		if(save_camera_planes)
		{

            for(int j=0;j<imgs_features[i]->pkeypoints.size();j++)
            {
                if(imgs_features[i]->pkeypoints[j]->cloud_p != NULL)
                {
                px_pos(0,0) = ((imgs_features[i]->pkeypoints[j]->pt.x - Kvec[i].at<double>(0,2))/Kvec[i].at<double>(0,0)); px_pos(1,0) = ((imgs_features[i]->pkeypoints[j]->pt.y - Kvec[i].at<double>(1,2))/Kvec[i].at<double>(1,1)); px_pos(2,0) = 1.0; //position of the projection in camera coord system, considering unit focal length z=1.0
                wd_pos = (R.inv()*px_pos) + c; // projection location in world's coordinate

                int R = (int)imgs_features[i]->pkeypoints[j]->R;
                int G = (int)imgs_features[i]->pkeypoints[j]->G;
                int B = (int)imgs_features[i]->pkeypoints[j]->B;

                (R+60<=255)? R+=60:R=255;
                (G+60<=255)? G+=60:R=255;
                (B+60<=255)? B+=60:B=255;


                ss<<wd_pos.at<double>(0,0)<<" "<<wd_pos.at<double>(1,0)<<" "<<wd_pos.at<double>(2,0)<<" "<< R<<" "<< G <<" "<< B <<endl;//" 255 255 0"<<endl;
                num_points_out++;
                }
            }

        }
	}
	else
	fprintf(fcamera,"%.12f %.12f %.12f\n",0.0,0.0,0.0); //camera not estimated

}

fclose(fcamera);
/////////////////////mean rotation/////////////////

 R = get_mean_rotation(imgs_features);

t(0,0) = 0; t(1,0)=0; t(2,0)=0;

 c = -R.inv()*t; // camera center in world's coordinate

 int i=5;

 for(int j=0;j<imgs_features[i]->pkeypoints.size();j++)
		{
			if(imgs_features[i]->pkeypoints[j]->cloud_p != NULL)
			{
            px_pos(0,0) = ((imgs_features[i]->pkeypoints[j]->pt.x - Kvec[i].at<double>(0,2))/Kvec[i].at<double>(0,0)); px_pos(1,0) = ((imgs_features[i]->pkeypoints[j]->pt.y - Kvec[i].at<double>(1,2))/Kvec[i].at<double>(1,1)); px_pos(2,0) = 1.0; //position of the projection in camera coord system, considering unit focal length z=1.0
            wd_pos = (R.inv()*px_pos) + c; // projection location in world's coordinate

            int R = (int)imgs_features[i]->pkeypoints[j]->R;
            int G = (int)imgs_features[i]->pkeypoints[j]->G;
            int B = (int)imgs_features[i]->pkeypoints[j]->B;

            (R+60<=255)? R+=60:R=255;
            (G+60<=255)? G+=60:R=255;
            (B+60<=255)? B+=60:B=255;

            if(save_camera_planes)
            {
                ss<<wd_pos.at<double>(0,0)<<" "<<wd_pos.at<double>(1,0)<<" "<<wd_pos.at<double>(2,0)<<" "<< "255"<<" "<< "255" <<" "<< "255" <<endl;//" 255 255 0"<<endl;
                num_points_out++;
            }

			}
		}


    export_points_wGPS(Cloud,R,gps_data);

////////////////////////////////////////////////////



for(int i=0;i<Cloud.size();i++)
{

	if(Cloud[i]->is_in_cloud && Cloud[i]->owners.size() > 2)
	{

		int c_cam = 0;

		for(int j=0;j<Cloud[i]->owners.size();j++)
			if(Cloud[i]->owners[j]->img_owner->triangulated)
				c_cam++;

		if(c_cam>=3)
		{

			ss<<Cloud[i]->x<<" "<<Cloud[i]->y<<" "<<Cloud[i]->z<<" "<<(int)Cloud[i]->owners[0]->R<<" "<<(int)Cloud[i]->owners[0]->G<<" "<<(int)Cloud[i]->owners[0]->B<<endl;
			num_points_out++;

			fprintf(fpoints,"%.12f %.12f %.12f\n",Cloud[i]->x,Cloud[i]->y,Cloud[i]->z);
		}
	}

}

fclose(fpoints);

FILE *f = fopen("/home/vnc/Dropbox/Projetos/PLYs/bundle_out.ply", "w");


    /* Print the ply header */
    fprintf(f, ply_header, num_points_out);
	fprintf(f,ss.str().c_str());

	fclose(f);

	//fprintf(f, "%0.16e %0.16e %0.16e 0 255 0\n", c[0], c[1], c[2]);

}
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc,char* argv[])
{
	int qualDescritor = 0; // 0 - sift / 1 - surf / 2 - orb / 3 - orb_pyramid.
	double down_factor = 1;  //resize img factor //0.3
	double meters_threshold = 140; // to cut the graph //40 //60
	double inliers_threshold = 200; //to cut bad edges
	double good_matches_percentile = 0.25;//percentile of best matches to hold.
	bool bUniqueCamera = false;
	bool show_final_result = false;
	bool debug = false;
	bool draw_bline_pair = true;
	bool use_undistort = false;
	bool use_flann = true;
	bool use_local_optima = false; //BA local_dont use it yet
	int skip_ba = 7;
	int rm_views_lessthan = 3;
	bool use_keypoint_reduction = false;
	char * path_to_imgs  = new char[512];
	char *params_file = new char[512];
    ParallelBA::DeviceT device  = ParallelBA::PBA_CPU_DOUBLE;
	int last_triangulated=-1;
	int use_ba = 5; //cvsba =2, opencv BA = 3, parallel_ba = 4, local_cvsba = 5
	float median_homography;
	char * output_path = new char[512];//"/home/itv/Sources/runs/run_small_mine/result";
	double ss_w,ss_h,focall;
	int LOCAL_WSIZE;

	confidence.first = 15.0; confidence.second=1;

	//string img_dir = "/home/itv/Sources/small_mine_sample";
	//  /home/itv/Sources/runs/run_small_mine
	srand(time(NULL));

    vector<Mat> imgs;
    vector<string> imgs_name;
    Mat G; //Graph
    vector<Point2d> gps_data;
    vector<Point3d> gps_3d;
    vector<vector<pairMatch*> > pairs;
    vector<imgMatches*> imgs_features;
    vector<cloudPoint*> Cloud;
    string opt;
    
    //cout<<"1 - build epipolar graph | 2 - run SfM"<<endl;
    //cout<<"int is: "<<sizeof(int)<<endl;
    //cin>>opt;

	//cin>>params_file;
	strcpy(params_file,argv[1]);
	opt = string(argv[2]);

	vector<double> flen;

	 read_params(down_factor,meters_threshold,inliers_threshold,good_matches_percentile,bUniqueCamera,use_flann,skip_ba,rm_views_lessthan,path_to_imgs,params_file,ss_w,ss_h,focall,flen);

	sprintf(output_path,"%s/result",path_to_imgs);
	string img_dir = path_to_imgs;

	LOCAL_WSIZE = skip_ba; //in case of LBA, skip_ba is the window size.

    //gettimeofday(&start, NULL);

	distortion = (cv::Mat_<double>(1,5) << 0, 0, 0, 0, 0); // initialize distortion coeffs vector 

    string filename =path_to_imgs; filename+="/out_camera_fisheye.xml";

	//camera parmeters

	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	// read camera matrix and distortion coefficients from file
	//Mat K;////, dis;
	if(fs.isOpened())
	{
	fs["camera_matrix"] >> K;
	fs["distortion_coefficients"] >> distortion;

	//for(int i=0;i<K.rows-1;i++)
	// for(int j=0;j<K.cols;j++)
      // K.at<double>(i,j)=K.at<double>(i,j)*down_factor;


	}

	if(opt=="1")
	open_imgs_dir(path_to_imgs,imgs_features,imgs,imgs_name,down_factor,distortion,use_undistort,K,qualDescritor);
	else
	{
	load_img_list(img_dir,imgs_name);

	}


         //cout<<"Focal length:"<<K.at<double>(0,0)<<endl;


    if(opt=="1")
    {
     thread_features(imgs_features, imgs, img_dir, imgs_name); //PARALLEL FEATURE EXTRACTION FOR ALL IMAGES
     save_imgMatches(imgs_features, img_dir, imgs_name); // Save the keypoints
 	}
 	else
 	{
 		load_imgMatches(imgs_features, img_dir, imgs_name); //Load keypoints

 			if(FISHEYE) //update calibration matrix
 			{
 			Mat newCamMat, map1, map2;
 			Size imageSize(imgs_features[0]->cols,imgs_features[0]->rows);
 			fisheye::estimateNewCameraMatrixForUndistortRectify(K, distortion, imageSize,
                                                              Matx33d::eye(), newCamMat, 1.0,imageSize,0.8); //1
          	fisheye::initUndistortRectifyMap(K, distortion, Matx33d::eye(), newCamMat, imageSize,
                                           CV_16SC2, map1, map2);

          	K = newCamMat;
      		}
 	}

	if(!fs.isOpened())
	{
         K = getK(focall,ss_w,ss_h,imgs_features[0]->cols, imgs_features[0]->rows); // construct cam matrix given camera params 20.0 17.3 13.0   4.5, 5.76, 4.29   4.3, 6.16, 4.62
        
    }

     load_camera_db(); //LOAD TXT FILE WITH SENSOR WIDTH DATABASE

    /* for(int i=0;i<camera_database.size();i++)
     {
     	cout<<camera_database[i].make<<" ";
     	cout<<camera_database[i].model<<" ";
     	cout<<camera_database[i].sensor_w<<endl; getchar();

     }
	*/

    bool update_focal= false;
    if(flen.size()>0)
    update_focal = true;

	int cont_exif = 0;

    for(int i=0;i<imgs_name.size();i++) //init intrinsics for each camera
    {
        Dvec.push_back((cv::Mat_<double>(1,5) << 0, 0, 0, 0, 0));
        Kvec.push_back(K.clone());
        double tt= 1.0;

	  //mount_K_EXIF(Kvec[i], img_dir+"/"+imgs_name[i],imgs[i]);

        if(fs.isOpened()) //has callib file opencv
        {
        Kvec[i].at<double>(0,0) = (Kvec[i].at<double>(0,0)/tt)*down_factor;
        Kvec[i].at<double>(1,1) = (Kvec[i].at<double>(1,1)/tt)*down_factor;
        Kvec[i].at<double>(0,2) = (Kvec[i].at<double>(0,2)/tt)*down_factor;
        Kvec[i].at<double>(1,2) = (Kvec[i].at<double>(1,2)/tt)*down_factor;

              //it is a callibrated dataset, set to had_intrinsics = true
        imgs_features[i]->had_intrinsics = true;

        //cout<<"Kvec= "<<Kvec[i]<<endl;
        }
		else if(update_focal) //has callib file 2. (txt list)
        {
        double scale_factor;
        cv::Mat m_ = open_cv_image(img_dir +"/" +imgs_name[i],down_factor, scale_factor);

        Kvec[i].at<double>(0,0) = flen[i]*scale_factor; Kvec[i].at<double>(1,1) = flen[i]*scale_factor;
        Kvec[i].at<double>(0,2) = imgs_features[i]->cols/2.0 -1; Kvec[i].at<double>(1,2) = imgs_features[i]->rows/2.0 -1;
        //cout<<flen[i]<<endl; getchar();
        imgs_features[i]->had_intrinsics = true;
        }
        else if(focall!=0) //has callib in sfm_params.txt
        {
            imgs_features[i]->had_intrinsics = true;
        }

        else //doesn't have any callib: Try to recover it from EXIF tags
        {

        //read focal length in exif


	        if(mount_K_EXIF(Kvec[i], img_dir+"/"+imgs_name[i], imgs_features[i])) //we recovered focal length (pixels) in EXIF
			{
				double f_ratio = 3.0;

				if( Kvec[i].at<double>(0,0) < ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)/f_ratio )
				{
					Kvec[i].at<double>(0,0) = ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)/f_ratio;
					Kvec[i].at<double>(1,1) = ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)/f_ratio;
				}
				else if (Kvec[i].at<double>(0,0) > ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)*f_ratio)
				{
					Kvec[i].at<double>(0,0) = ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)*f_ratio;
					Kvec[i].at<double>(1,1) = ((double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2)*f_ratio;
				}
				else
				{
					imgs_features[i]->EXIF = true;
	        		cont_exif++;
	        		if (cont_exif == 1)
	        			cout<<"Using estimated Kmatrix from EXIF = "<<endl<<Kvec[i]<<endl;
				}


	        }
	        else //assuming x axis generic FoV 50.0
	        {
	        	Kvec[i].at<double>(0,0) = (double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2;
	        	Kvec[i].at<double>(1,1) = (double)max(imgs_features[i]->rows,imgs_features[i]->cols)*1.2;
	        }


	      // update cx,cy

	        Kvec[i].at<double>(0,2) = (imgs_features[i]->cols/2.0 - 1.0);
	        Kvec[i].at<double>(1,2) = (imgs_features[i]->rows/2.0 - 1.0);
        }
    }

    cout<<cont_exif<<" images had focal length in EXIF, in a total of "<<imgs_features.size()<<endl; //getchar();///

    if(bUniqueCamera)
    	K = Kvec[0].clone();

     D = (cv::Mat_<double>(1,5) << 0, 0, 0, 0, 0);

    /*cout<<"pair 109 k = " << Kvec[109]<<endl;
    cout<<"pair 368 k = " << Kvec[368]<<endl;

    cout<<"img name 1 = " << imgs_name[109]<<endl;
    cout<<"img name 2 = "<< imgs_name[368]<<endl; getchar();
    */

    if(fs.isOpened())
    	fs.release();

	//G.create(imgs.size(),imgs.size(),CV_64F);
    G = Mat::zeros(imgs_name.size(),imgs_name.size(),CV_64F);
	double lat, longit;
    long cont_m=0;
	//cout<<imgs_name[0]<<endl;
	for(int i=0;i<imgs_name.size();i++)
	{
	     double alt = read_exif_gps(lat,longit,img_dir+"/"+imgs_name[i]);
	     Point2d latlong(lat,longit);
	     Point3d latlonalt(lat,longit,alt);
	     if(latlong.x ==0 || latlong.y ==0)
            cont_m++;
         // getchar();
	     gps_data.push_back(latlong);
	     gps_3d.push_back(latlonalt);

	}
  //printf(" distance = %f\n",distFrom(gps_data[0].x,gps_data[0].y,gps_data[1].x,gps_data[1].y));
    cout<<"Images without GPS: " <<cont_m<<endl; //getchar();
    save_GPS_list(gps_3d,img_dir+"/result/"); //getchar();


   /*  for(int i=0;i<imgs.size();i++) // features extraction for all images
    {
    imgMatches *im = new imgMatches;
    compute_keypoints(im,imgs[i],qualDescritor);
    im->id = i;

    imgs_features.push_back(im);
    cout<<"Done extracting feature for image " <<i+1<<endl;
   // cout<<imgs_features[i]->keypoints_img.size()<<endl;
    }*/


//    thread_features(imgs_features, imgs); //PARALLEL FEATURE EXTRACTION

    if(opt=="1")
    {

    generate_I_Graph(G,imgs_features); //Generate initial graph considering the vocab tree
     
     //for(int i=0;i<G.rows;i++)
     //	for(int j=0;j<G.cols;j++)
     //		G.at<double>(i,j) = 1.0;

    generate_D_Graph(G,gps_data);

    cut_D_Graph(G, meters_threshold);
    cout<<"Number of total pairs: "<<pair_count<<endl;
    cout<<"Pair ratio: "<<pair_count/(double)imgs.size()<<endl<<endl;
	}

    //creating pairMatch array
    for(int i=0;i<imgs_features.size();i++)
    {
        pairs.push_back(vector<pairMatch*>());
         for(int j=0;j<imgs_features.size();j++)
         {
            pairMatch *p_p;
            pairs[i].push_back(p_p);
         }
    }



	//fs["Distortion_Coefficients"] >> dis;
	// close the input file
	fs.release();


	 if(G.rows<=30)
    	 cout<<G<<endl;
    //getchar();


////////////////////////////////////////////Matching Pairs///////////////////////////////

     if(opt=="1")
    {
    match_pair_parallel(G,pairs,imgs_features,Cloud,inliers_threshold);

   /* for(int i=0;i<G.rows;i++)   //calculating F_mats and inliers
     for(int j=i+1; j<G.cols; j++)
     {
         if(G.at<double>(i,j) == 1) // there's an edge connecting the two images
         {
             vector<Point2f> cam_1;
             vector<Point2f> cam_2;
             vector<long>idx_cam_1, idx_cam_2;
             int w_homog_inliers=0;

             Mat F_mat;
             pairMatch *pm = new pairMatch;

            if(false)
           	;// G.at<double>(i,j) = (double)match_pair(imgs_features[i], imgs_features[j],F_mat,qualDescritor, cam_1, cam_2,idx_cam_1,idx_cam_2, Cloud,imgs[i],imgs[j],w_homog_inliers,use_flann,good_matches_percentile,inliers_threshold);
            else
            {

            cv::Mat Maux;
            Maux.create(1,1,CV_64F);
            G.at<double>(i,j) = (double)match_pair(imgs_features[i], imgs_features[j],F_mat,qualDescritor, cam_1, cam_2,idx_cam_1,idx_cam_2, Cloud,imgs[i],imgs[j],w_homog_inliers,use_flann,good_matches_percentile,inliers_threshold,i,j);
            }
            pm->cam1 = cam_1;
            pm->cam2 = cam_2;
            pm->idx_cam1 = idx_cam_1;
            pm->idx_cam2 = idx_cam_2;


            pm->F_mat = F_mat;



            pm->triangulated = false;
            pm->h_tested = false;
            pm->homog_inliers = w_homog_inliers;

            pairs[i][j] = pm;

            if(cam_1.size()==0 || cam_2.size()==0)
            {
                G.at<double>(i,j) = 0; //There's no way this pair could me matched
            }
         }
     }

     */
   //////////////////////////////////////END OF MATCHING PAIRS//////////////////////////////////////////

   /* vector<float> median;

    for(int i=0;i<G.rows;i++)   //calculating homography median
     for(int j=i+1; j<G.cols; j++)
     {
       if(G.at<double>(i,j))
       median.push_back(pairs[i][j]->homog_inliers);
     }
     sort(median.begin(),median.end());
     median_homography = median[(int)median.size()/2];
    */

     cout<<"Building maximum spanning tree-based Epipolar Graph..."<<endl;
    vector<pair<int,int> > prune_g = get_Kruskal(G); //returns the elements that are NOT in Maximal spanning tree
    inliers_threshold += 60; //120
    

    /*for(int i=0;i<G.rows;i++)
    	for(int j=i+1;j<G.cols;j++)
    	{
    		if(G.at<double>(i,j) < inliers_threshold)
       		 {
          		  G.at<double>(i,j) = 0;
       		 }
    	}
	*/
    	
    for(int i=0;i<prune_g.size();i++) //pruning graph if the edge is lower than a threshold
    {
        if(G.at<double>(prune_g[i].first,prune_g[i].second) < inliers_threshold)
        {
            G.at<double>(prune_g[i].first,prune_g[i].second) = 0;
        }
    }
    
	
    cout<<"Done filtering the epipolar graph..."<<endl;

   /***********Saving Epipolar Graph and Matches****************/
    save_epgraph(G, img_dir, imgs_name);
    save_pairMatch(pairs, img_dir, imgs_name, G);
    save_img_list(img_dir,imgs_name);

    //TODO: let the user continue the reconstruction without the need to restart the program. -- DEALLOCATE ALL NON-REQUIRED STRUCTURES --
     Kvec.clear();
     Dvec.clear();
     K.release();
     D.release();
     distortion.release();

	return 0;

	}
	else
	{
	 /***********Loading Epipolar Graph and Matches****************/

		load_epgraph(G, img_dir, imgs_name);
		//cout<<"Teste loading:"<<endl<<G<<endl;

		load_pairMatch(pairs, img_dir, imgs_name, G);


		/*********Updating the graph by the largest connected component********/


	}


	/************building 2d-3d projections structure************/
	vector<pair<int,int> > all_pairs;

	    for(int i=0;i<G.rows;i++)
         for(int j=i+1; j<G.cols; j++)
         {
             if(G.at<double>(i,j) > 0)
             	all_pairs.push_back(make_pair(i,j));
         }

    cout<<"Done pairwise matching. Now associating 3D points with projections..."<<endl;
	build_projections_BFSearch(G,pairs,imgs_features,all_pairs,inliers_threshold,Cloud); //initialize 3d structure according to the 2D  projs association
  	cout<<"Done associating 3D points. Now filling 3D data-structure..."<<endl;
	fill_pt3d_vector(imgs_features,Cloud); //initialize Cloud vector with the 3d points found
    cout<<"Done filling data-structure."<<endl;

    cout<<"Missing projections: "<<test_imgfeatures(imgs_features)<<endl; //getchar();

	remove_inconsistent_projs(imgs_features);




   //int contd=0;

    /*
    for(long i=0;i<Cloud.size();i++)
    {
        if(Cloud[i]->owners.size()< rm_views_lessthan)
        {//removing points with only n correspondences
           for(int j=0;j<Cloud[i]->owners.size();j++)
           {
               Cloud[i]->owners[j]->cloud_p = NULL;
           }
            Cloud[i]->owners.clear();
          delete(Cloud[i]);
          Cloud.erase(Cloud.begin()+i); i--;


        }
    }
    */
   cout<<"Now filtering 3D points with projections fewer than "<<rm_views_lessthan<<endl;
 //remove_projs_lessthan(Cloud,rm_views_lessthan);

    /*for(int i=0;i<G.rows;i++)
     for(int j=i+1; j<G.cols; j++)
     {
        for(int k=0;k<pairs[i][j]->idx_cam1.size();k++) //removing all correspondences < 2 views
           {
               if(imgs_features[i]->pkeypoints[pairs[i][j]->idx_cam1[k]]->cloud_p ==NULL) //remove
               {
                   pairs[i][j]->idx_cam1.erase(pairs[i][j]->idx_cam1.begin()+k);
                   pairs[i][j]->idx_cam2.erase(pairs[i][j]->idx_cam2.begin()+k);
                   pairs[i][j]->cam1.erase(pairs[i][j]->cam1.begin()+k);
                   pairs[i][j]->cam2.erase(pairs[i][j]->cam2.begin()+k);
                   k--;

               }
           }
    }*/

    //cout<<"qtd owners: "<<contd<<endl;

    if(G.rows<=30)
   	 cout<<"pruned G "<<G<<endl;


 pair<int,int> highest_inlier;
 vector<double> reproj_error;
 long highest_value;
 bool baseline_done=false;
 double lowest_reprojerr=9999;
 double current_reprojerr;

//do{
 highest_value=-1;

/*
  for(int i=0;i<G.rows;i++)   //find the best pair for baseline triangulation
     for(int j=i+1; j<G.cols; j++)
     {

        if(G.at<double>(i,j)  )
        {


          imgs_features[i]->P_mat = Matx34d (1,0,0,0,
                                             0,1,0,0,
                                             0,0,1,0);

         find_P1(imgs_features[j]->P_mat, pairs[i][j]->F_mat,pairs[i][j]->cam1, pairs[i][j]->cam2,reproj_error,i,j);
         current_reprojerr = reproj_error[(int)(reproj_error.size()*0.75)]; //reproj_error[reproj_error.size()/2];//mean(reproj_error)[0];

        if( G.at<double>(i,j) > 450)//inliers_threshold + 150)
         if(current_reprojerr < lowest_reprojerr ) //&& !pairs[i][j]->h_tested)
         {
             lowest_reprojerr = current_reprojerr ;
             highest_inlier.first = i; highest_inlier.second = j;
             highest_value=G.at<double>(i,j);
         }

        }
     }

     //pairs[highest_inlier.first][highest_inlier.second]->h_tested = true;
     cout<<"Inliers reprjerror: " << highest_value <<endl;
*/
     cout<<"Finding best candidates to the seed reconstruction..."<<endl;
    find_good_baseline_pair(G,imgs_features,pairs,highest_inlier);
    cout<<"Found best candidate."<<endl;


    /***Drawing matches of the Baseline Pair***/
    vector<DMatch> matches_b;
    vector<KeyPoint> kp_im1;
    vector<KeyPoint> kp_im2;
    cv::Mat match_baseline;
    cv::Mat p1_img, p2_img;


    if(draw_bline_pair)
    {

	    p1_img = open_cv_image(img_dir+"/result/undistorted/"+imgs_name[highest_inlier.first],down_factor);
	    p2_img = open_cv_image(img_dir+"/result/undistorted/"+imgs_name[highest_inlier.second],down_factor);

	    for(int i=0;i<pairs[highest_inlier.first][highest_inlier.second]->cam1.size();i++)
	    {
	        KeyPoint kp_1;
	        KeyPoint kp_2;
	        DMatch m_base;

	        m_base.queryIdx = i;
	        m_base.trainIdx = i;

	        kp_1.pt = pairs[highest_inlier.first][highest_inlier.second]->cam1[i];
	        kp_2.pt = pairs[highest_inlier.first][highest_inlier.second]->cam2[i];

	        kp_im1.push_back(kp_1);
	        kp_im2.push_back(kp_2);

	        matches_b.push_back(m_base);
	    }

	     drawMatches(p1_img, kp_im1, p2_img, kp_im2,
	                matches_b, match_baseline, Scalar::all(-1), Scalar::all(-1),
	                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	    imwrite("/home/vnc/Dropbox/Projetos/PLYs/baseline_match.jpg",match_baseline);
	}

     imgs_features[highest_inlier.first]->P_mat = Matx34d (1,0,0,0,
                                                          0,1,0,0,
                                                          0,0,1,0);

    double found_median=0;
    int n_of_triangulations=0;
    vector<bool> inliers;

    if(find_P1(imgs_features[highest_inlier.second]->P_mat, pairs[highest_inlier.first][highest_inlier.second]->F_mat,pairs[highest_inlier.first][highest_inlier.second]->cam1, pairs[highest_inlier.first][highest_inlier.second]->cam2,reproj_error,highest_inlier.first,highest_inlier.second,inliers))
       {
           cout<<"best pair found, 0.75 median = "<< reproj_error[(int)(reproj_error.size()*0.75)] <<endl;
           found_median=reproj_error[(int)(reproj_error.size()*0.75)];
           cout<<"Essential mat ="<<endl<<pairs[highest_inlier.first][highest_inlier.second]->F_mat<<endl;
           cout<<"P1 = "<<endl<<imgs_features[highest_inlier.second]->P_mat<<endl;
           baseline_done = true;
           cout<<"img 1 = "<<imgs_name[highest_inlier.first]<<endl;
           cout<<"img 2 = "<<imgs_name[highest_inlier.second]<<endl;

       }


//}while(!baseline_done && highest_value!=-1);

/*
Mat kp = K*Mat(imgs_features[highest_inlier.first]->P_mat);
Mat kp1 = K*Mat(imgs_features[highest_inlier.second]->P_mat);

for(int i=0;i<kp.rows;i++)
 for(int j=0;j<kp.cols;j++)
 {
  imgs_features[highest_inlier.first]->P_mat(i,j) = kp.at<double>(i,j);
  imgs_features[highest_inlier.second]->P_mat(i,j)= kp1.at<double>(i,j);
 }
cout<<"A  A "<<  imgs_features[highest_inlier.first]->P_mat<<endl;
cout<<"A  A "<<  imgs_features[highest_inlier.second]->P_mat<<endl;
*/

 vector<Point3d> triangulated_cloud; //triangulating points
 vector<bool> inliers_triangulation;


 TriangulatePointsRectified(imgs_features[highest_inlier.second]->P_mat,imgs_features[highest_inlier.first]->P_mat, pairs[highest_inlier.first][highest_inlier.second]->cam1, pairs[highest_inlier.first][highest_inlier.second]->cam2,highest_inlier.first,highest_inlier.second, triangulated_cloud,reproj_error,inliers_triangulation);
 pairs[highest_inlier.first][highest_inlier.second]->triangulated = true;
 imgs_features[highest_inlier.first]->triangulated=true;
 imgs_features[highest_inlier.second]->triangulated=true;
 imgs_features[highest_inlier.first]->estimated=true;
 imgs_features[highest_inlier.second]->estimated=true;

 vector<bool> incosistencies(imgs_features[highest_inlier.first]->pkeypoints.size(),false);
 int n_inc = 0;
 int n_out_triang = 0;
 int n_null_pts = 0;

found_median=reproj_error[(int)(reproj_error.size()*0.5)];
for(int i=reproj_error.size()*0.94;i<reproj_error.size();i++)
           	cout<<reproj_error[i]<<endl;

 for(long i=0;i<triangulated_cloud.size();i++)
 {

     cloudPoint * cp = imgs_features[highest_inlier.first]->pkeypoints[pairs[highest_inlier.first][highest_inlier.second]->idx_cam1[i]]->cloud_p;
     int idx = pairs[highest_inlier.first][highest_inlier.second]->idx_cam1[i];

    if(cp !=NULL && inliers_triangulation[i])
    {
    	 if(incosistencies[idx])
    	 	n_inc++;

    	 incosistencies[idx] = true;

	     cp->is_in_cloud = true;
	     cp->x = triangulated_cloud[i].x;
	     cp->y = triangulated_cloud[i].y;
	     cp->z = triangulated_cloud[i].z;

	     n_of_triangulations++;

    }
    else if(!inliers_triangulation[i] && cp != NULL)
    	n_out_triang++;
    else if (inliers_triangulation[i] && cp==NULL)
    	n_null_pts++;


 }

cout<<"N. of incosistencies: "<<n_inc<<endl;
cout<<"N. of triagulation outliers: "<<n_out_triang<<endl;
cout<<"N. of null points: "<<n_null_pts<<endl;

vector<long> cam_test = pairs[highest_inlier.first][highest_inlier.second]->idx_cam1;

std::sort(cam_test.begin(),cam_test.end());

for(int i=0;i<cam_test.size()-1;i++)
{
	if(cam_test[i] == cam_test[i+1])
		cout<<"Duplicate index in idx_cam1: "<<i<<" "<<i+1<<" : "<<cam_test[i]<<endl;
}

cout<<"Best pair re-triangulation median = "<<found_median<<endl;
cout<<"Number of points updated in cloud = "<<n_of_triangulations<<endl;
cout<<"number of projs = "<<pairs[highest_inlier.first][highest_inlier.second]->cam1.size()<<endl;

//getchar();

    vector<int> vet_idx;
    vector<int> idx_local_opt;
    vector<int> idx_global_opt;

    idx_local_opt.push_back(highest_inlier.first);
    idx_local_opt.push_back(highest_inlier.second);

    idx_global_opt.push_back(highest_inlier.first);
    idx_global_opt.push_back(highest_inlier.second);

       // cout<<"foi";remove_outliersfromcloud(Cloud);
       /*remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);*/
    	remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
    	correct_cloud(Cloud);
    	
       // remove_outliers_by_reprojerror_2(imgs_features,K,imgs_features[0]->cols,Cloud);


        ///////BUNDLE ADJUSTMENT BASELINE PAIR////////
         Ceres_Bundler BA_ceres;
         cv::Mat new_k = BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_local_opt,Cloud,true,true,false,bUniqueCamera,0); //global BA

         if(new_k.rows > 0)
         {
	       K.at<double>(0,0) = new_k.at<double>(0,0);
	       K.at<double>(1,1) = new_k.at<double>(0,0);

	       D.at<double>(0,0) = new_k.at<double>(0,1);

	       //cout<<K<<endl; getchar();
	       //cout<<D<<endl; getchar();

         }

         ba_counter=2;


        //cout<<"foi";remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
 //stereo_Rectify(imgs[highest_inlier.first],imgs[highest_inlier.second],imgs_features[highest_inlier.first]->P_mat, imgs_features[highest_inlier.second]->P_mat); // finds the disparity map for the baseline triang after ba

  //  if(show_baseline_triang)
   // show_pointCloud(Cloud); // showing baseline triang

    long qtd_matches;
    int best_img=0;
    bool all_triangulated=false;

    while(ba_counter <= imgs_features.size() ) //!all_triangulated)
    {

        highest_value = -1;
        all_triangulated=true;
        for(long i=0;i<imgs_features.size();i++)   //find image with highest 2d-3d matches
        {
            qtd_matches =0;

            if(!imgs_features[i]->estimated && !imgs_features[i]->already_rmv)
            { //if(i==0) cout<<"verificando img " <<i<<" "<<imgs_features[i]->triangulated<<endl;

                for(long j=0;j<imgs_features[i]->pkeypoints.size();j++)
                {
                    if(imgs_features[i]->pkeypoints[j]->cloud_p!= NULL && imgs_features[i]->pkeypoints[j]->cloud_p->is_in_cloud)
                    {
                        qtd_matches++;
                    }
                }

                if(qtd_matches>highest_value)
                {
                    highest_value = qtd_matches;
                    best_img = i;
                }
                all_triangulated=false;
            }
         }

        if(all_triangulated || highest_value<32) //3 //20 //60
        {

        if(highest_value<32)
        cout<<"Did not find enough 2d-3d correspondences. Finishing the reconstruction...";

        break;
        }

         vector<Point2f> points2d;
         vector<Point3f> points3d;
         vector<long> in_outliers_idx;
//
         for(long i=0;i<imgs_features[best_img]->pkeypoints.size();i++)
         {
              if(imgs_features[best_img]->pkeypoints[i]->cloud_p!= NULL && imgs_features[best_img]->pkeypoints[i]->cloud_p->is_in_cloud)
              {
                  points2d.push_back(imgs_features[best_img]->pkeypoints[i]->pt);
                  Point3f p3d(imgs_features[best_img]->pkeypoints[i]->cloud_p->x,imgs_features[best_img]->pkeypoints[i]->cloud_p->y,imgs_features[best_img]->pkeypoints[i]->cloud_p->z);
                  points3d.push_back(p3d);
                  in_outliers_idx.push_back(i);
              }
         }

         Mat vecr, tmp_dist_coeff;
         cv::Mat_<double> t;
         cv::Mat_<double> R;
         vector<int> inliers;



		double minVal,maxVal; //cv::minMaxIdx(points2d,&minVal,&maxVal);
		//solvePnPRansac(points3d, points2d, K, tmp_dist_coeff, vecr, t, false, 3000, 0.0035 * maxVal, 0.25 * (double)(points2d.size()), inliers, CV_ITERATIVE); //EP_NP //0.006 //1000 //3000
        //solvePnPRansac(points3d, points2d, K, tmp_dist_coeff, vecr, t, false, 5000, 0.0035 * (double)imgs[0].rows,  (double)(points2d.size()), inliers, CV_ITERATIVE); //EP_NP //0.006 //1000 //3000

        //0.0068
         //solvePnPRansac(points3d, points2d, Kvec[best_img], Dvec[best_img], vecr, t, false, 2000, 0.0068 * (double)imgs[0].rows,  0.97*(double)(points2d.size()), inliers, CV_ITERATIVE);

         Matx34d newProjM;
         double ac_threshold=16.0;

         //getchar();

         if(ACONTRARIO)
        {

        if(!bUniqueCamera)
        	openMVG::sfm::SolvePnP_AC_RANSAC(imgs_features[best_img]->cols, imgs_features[best_img]->rows, points2d, points3d, Kvec[best_img],Dvec[best_img], newProjM, ac_threshold,true);//imgs_features[best_img]->had_intrinsics);
		else
		{
			openMVG::sfm::SolvePnP_AC_RANSAC(imgs_features[best_img]->cols, imgs_features[best_img]->rows, points2d, points3d, K, D, newProjM, ac_threshold,true);//imgs_features[best_img]->had_intrinsics);
			Kvec[best_img] = K.clone();
			Dvec[best_img] = D.clone();
		}

		imgs_features[best_img]->estimated = true;
		imgs_features[best_img]->P_mat = newProjM;
        cout<<">>>>>>>>Solved PnP "<<best_img << " with [" <<points3d.size()<<"] correspondences and threshold "<< ac_threshold <<": ---> Progress: " <<ba_counter /*++pnp_count*/ <<"/"<<  imgs_features.size()<< endl; //
		cout<<"image: "<<imgs_name[best_img]<<endl;
		}
		else //OpenCV's solve PnP
		{
			ac_threshold = 8.0;
			vector<double> d_coeffs_v;

			if(!bUniqueCamera)
				solvePnPRansac(points3d, points2d, Kvec[best_img], Dvec[best_img], vecr, t, false, 2048, ac_threshold, 0.999, inliers, CV_ITERATIVE);
			else
				{
					for(int m=0;m<5;m++)
						d_coeffs_v.push_back(D.at<double>(0,m));

					cout<<"Solving PnP OpenCV with dist coeffs: "<<d_coeffs_v[0]<<" "<<d_coeffs_v[1]<<" "<<d_coeffs_v[2]<<" "<<d_coeffs_v[3]<<" "<<d_coeffs_v[4]<<endl;

					//solvePnPRansac(points3d, points2d, K, d_coeffs_v, vecr, t, false,  CV_ITERATIVE); // 2048, ac_threshold, 0.999, inliers, 
					cv::solvePnPRansac(points3d, points2d, K, d_coeffs_v, vecr, t, false, 2048, ac_threshold, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
					Kvec[best_img] = K.clone();
					Dvec[best_img] = D.clone();

				}


			cout<<"Inliers reported by CV_RANSAC: "<<inliers.size()<<" of "<< points2d.size() <<endl;

			cout<<">>>>>>>>Solved PnP (OpenCV) "<<best_img << " with [" <<points3d.size()<<"] correspondences and threshold "<< ac_threshold <<": ---> Progress: " <<ba_counter<<"/"<<  imgs_features.size()<< endl;
			imgs_features[best_img]->estimated = true;
			Rodrigues(vecr, R);

       	 	imgs_features[best_img]->P_mat =  Matx34d ( R(0,0),R(0,1),R(0,2),t(0),
                        		 						R(1,0),R(1,1),R(1,2),t(1),
                       			 						R(2,0),R(2,1),R(2,2),t(2) );

       	 	/*Ceres_Bundler BA_ceres_cam;

      		 vector<int> adjust_cam; adjust_cam.push_back(best_img);
      	     BA_ceres_cam.Adjust(imgs_features,Kvec,Dvec,adjust_cam,Cloud,true,true,true,false,0); //Camera-Only Bundle Adjustment

      	     solvePnPRansac(points3d, points2d, Kvec[best_img], Dvec[best_img], vecr, t, false, 2048, ac_threshold, 0.999, inliers, CV_ITERATIVE);
			 cout<<">>>>>>>>Solved PnP (OpenCV) "<<best_img << " with [" <<points3d.size()<<"] correspondences and threshold "<< ac_threshold <<": ---> Progress: " <<ba_counter<<"/"<<  imgs_features.size()<< endl;
			 Rodrigues(vecr, R);

       	 	imgs_features[best_img]->P_mat =  Matx34d (R(0,0),R(0,1),R(0,2),t(0),
                        		 R(1,0),R(1,1),R(1,2),t(1),
                       			 R(2,0),R(2,1),R(2,2),t(2));

			*/

            cout<<"image: "<<imgs_name[best_img]<<endl; 
            //imgs_features[best_img]->triangulated = false;
		}

		/*if(ac_threshold > 24.0) //solve uncalibrated and estimate focal length from decomposition
		{
			openMVG::sfm::SolvePnP_AC_RANSAC(imgs_features[best_img]->cols, imgs_features[best_img]->rows, points2d, points3d, Kvec[best_img], newProjM, ac_threshold,false);
		}*/


		bool is_camera_good = true;
       /* for(long i=0;i<in_outliers_idx.size();i++) //setting all to false (removing outliers from pnp ransac)
        {
            imgs_features[best_img]->pkeypoints[in_outliers_idx[i]]->cloud_p->is_in_cloud = false;
        }

        for(long i=0;i<inliers.size();i++) //remove outliers from cloud
        {
          //  if(!inliers[i])
           // {
               imgs_features[best_img]->pkeypoints[in_outliers_idx[inliers[i]] ]->cloud_p->is_in_cloud =true;
            //}
        }
        */

       /* Rodrigues(vecr, R);

        imgs_features[best_img]->P_mat =  Matx34d (R(0,0),R(0,1),R(0,2),t(0),   //inverting signals to get camera pose in world instead of worlds coordinate system in cam's coord. system
                                                    R(1,0),R(1,1),R(1,2),t(1),
                                                    R(2,0),R(2,1),R(2,2),t(2));
        */



        double outlier_ratio = remove_outliers_from_cam(imgs_features[best_img],Kvec[best_img],Dvec[best_img],ac_threshold);
    

        cout<<"Percentage(%%) of outliers: "<<outlier_ratio*100.0<<endl; ////getchar();
        //getchar();

        if((outlier_ratio >= 0.75 || ac_threshold > 24.0) && !imgs_features[best_img]->had_intrinsics && ACONTRARIO) //bad a contrario estimation, try decompose and extract focal length
        {
        	cout<<"Too many outliers: trying to solve uncalibrated ..."<<endl<<endl;
        	/*solvePnPRansac(points3d, points2d, Kvec[best_img], Dvec[best_img], vecr, t, false, 5000, 0.0068 * (double)imgs_features[best_img]->cols, 0.9999, inliers, CV_ITERATIVE);

        	Rodrigues(vecr, R);

       	 	imgs_features[best_img]->P_mat =  Matx34d (R(0,0),R(0,1),R(0,2),t(0),   //inverting signals to get camera pose in world instead of worlds coordinate system in cam's coord. system
                                                    R(1,0),R(1,1),R(1,2),t(1),
                                                    R(2,0),R(2,1),R(2,2),t(2));
             */

			openMVG::sfm::SolvePnP_AC_RANSAC(imgs_features[best_img]->cols, imgs_features[best_img]->rows, points2d, points3d, Kvec[best_img], Dvec[best_img], newProjM, ac_threshold,false);
			imgs_features[best_img]->P_mat = newProjM;

       	 	outlier_ratio = remove_outliers_from_cam(imgs_features[best_img],Kvec[best_img],Dvec[best_img],0.0068*(double)imgs_features[best_img]->cols);

       	 	if(outlier_ratio >=0.75 || ac_threshold > 24.0)
       	 		is_camera_good = false;

        }
        else if(outlier_ratio >=0.75 && imgs_features[best_img]->had_intrinsics)
        {
            is_camera_good = false;
        }


        if(is_camera_good) // if we were able to estimate a consistent pose
        {
	        if(pnp_count==1) //first
	        {confidence.first=ac_threshold; confidence.second=1;}
	        else //if(ac_threshold<=35.0)
	        {confidence.first+=ac_threshold; confidence.second++;}

	    	imgs_features[best_img]->triangulated = true;
	        idx_local_opt.push_back(best_img);
	        idx_global_opt.push_back(best_img);



		         Ceres_Bundler BA_ceres_cam;

	      		 vector<int> adjust_cam; adjust_cam.push_back(best_img);
	      	     BA_ceres_cam.Adjust(imgs_features,Kvec,Dvec,adjust_cam,Cloud,true,true,true,bUniqueCamera,0); //Camera-Only Bundle Adjustment

	      		 cout<<"Adjusted cam for K= "<<endl<<Kvec[best_img]<<endl<<" and D= "<<Dvec[best_img]<<endl;

      		

      		 ba_counter++;

   		 }
   		 else
   		 	cout<<"Camera resectioning failed!"<<endl;




        //adjust_camera_cvsba(Kvec[best_img],Dvec[best_img],imgs_features[best_img],ac_threshold); //adjusts camera motion and intrinsics

       // printf("second %f %f\n",Dvec[best_img].at<double>(0,0),Dvec[best_img].at<double>(0,1));

        for(int i=1;best_img + i <G.cols || best_img - i>= 0;i++) //triangulating image to connected vertices
        {
           if( i+best_img < G.cols && G.at<double>(best_img,best_img+i) > 0  && imgs_features[best_img+i]->triangulated)
           {
              int first, second;
              first = best_img;
              second = best_img+i;
            //idx_local_opt.push_back(i);


           	 if(imgs_features[best_img]->triangulated)//pairs[first][second]->homog_inliers/(double)pairs[first][second]->cam1.size() <=0.9) //Check for non-parallaxing images
          	  {
	          		      TriangulatePointsRectified(imgs_features[second]->P_mat,imgs_features[first]->P_mat, pairs[first][second]->cam1, pairs[first][second]->cam2, first, second, triangulated_cloud,reproj_error,inliers_triangulation);
	           		      pairs[first][second]->triangulated = true;
	           		      imgs_features[first]->triangulated=true;
	            		  imgs_features[second]->triangulated=true;
	            		  vector<cloudPoint*> optimize;

	             	    for(long j=0;j<triangulated_cloud.size();j++)
	             	    {
	                     cloudPoint * cp = imgs_features[first]->pkeypoints[pairs[first][second]->idx_cam1[j]]->cloud_p;
	                     cloudPoint * cp2 = imgs_features[second]->pkeypoints[pairs[first][second]->idx_cam2[j]]->cloud_p;


		                     if(cp!= NULL && cp == cp2)
		                     {

		                     if(!cp->is_in_cloud && inliers_triangulation[j])
		                        {

		                         cp->is_in_cloud = true;
		                         cp->x = triangulated_cloud[j].x;
		                         cp->y = triangulated_cloud[j].y;
		                         cp->z = triangulated_cloud[j].z;
		                         cp->can_ba = 4;

		                         optimize.push_back(cp);

		                        }

		                     }

               			}

               	//Ceres_Bundler BA_ceres_cam;

      		 	//vector<int> adjust_cam; adjust_cam.push_back(first); adjust_cam.push_back(second);
      		 	//if(optimize.size() > 0)
      	    	//BA_ceres_cam.Adjust(imgs_features,Kvec,Dvec,adjust_cam,optimize,false,false,true,bUniqueCamera,0); //Get optimal triangulation



           		}


         }

          	if( best_img - i >=0 && G.at<double>(best_img-i,best_img) > 0  && imgs_features[best_img-i]->triangulated)
           {
              int first, second;
              first = best_img-i;
              second = best_img;
            //idx_local_opt.push_back(i); //nao


           	 if(imgs_features[best_img]->triangulated)//pairs[first][second]->homog_inliers/(double)pairs[first][second]->cam1.size() <=0.9) //Check for non-parallaxing images
          	  {
	          		     TriangulatePointsRectified(imgs_features[second]->P_mat,imgs_features[first]->P_mat, pairs[first][second]->cam1, pairs[first][second]->cam2, first, second, triangulated_cloud,reproj_error,inliers_triangulation);
	           		     pairs[first][second]->triangulated = true;
	           		     imgs_features[first]->triangulated=true;
	            		 imgs_features[second]->triangulated=true;
	            		 vector<cloudPoint*> optimize;

	             	    for(long j=0;j<triangulated_cloud.size();j++)
	             	    {
	                     cloudPoint * cp = imgs_features[first]->pkeypoints[pairs[first][second]->idx_cam1[j]]->cloud_p;
		                 cloudPoint * cp2 = imgs_features[second]->pkeypoints[pairs[first][second]->idx_cam2[j]]->cloud_p;


		                     if(cp!= NULL && cp == cp2)
		                     {

		                     if(!cp->is_in_cloud && inliers_triangulation[j])
		                        {

		                         cp->is_in_cloud = true;
		                         cp->x = triangulated_cloud[j].x;
		                         cp->y = triangulated_cloud[j].y;
		                         cp->z = triangulated_cloud[j].z;
		                         cp->can_ba = 4;

		                         optimize.push_back(cp);

		                        }

		                     }



               			}

               	//Ceres_Bundler BA_ceres_cam;

      		 	//vector<int> adjust_cam; adjust_cam.push_back(first); adjust_cam.push_back(second);
      		 	//if(optimize.size() > 0)
      	    	//BA_ceres_cam.Adjust(imgs_features,Kvec,Dvec,adjust_cam,optimize,false,false,true,bUniqueCamera,0); //Get optimal triangulation

           		}


         }


        }

        //imgs_features[best_img]->triangulated  = true;
      //  remove_outliersfromcloud(Cloud);
        /*all_triangulated = */  /*remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);*/
	    //remove_outliers_by_reprojerror_2(imgs_features,K,imgs_features[0]->cols,Cloud);


        ///////BUNDLE ADJUSTMENT////////

            //remove_outliersfromcloud(Cloud);

         cout<<"##########################################"<<endl;
         cout<<" @ Camera "<<ba_counter<<" done. "<<imgs_features.size() << " in total."<<endl;
         cout<<"##########################################"<<endl;

         if(ba_counter <= 25) // GLOBAL BA in the first 25 cameras always
         {
         	 cout<<"-----------------------------------------------"<<endl;
             remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
             correct_cloud(Cloud);
             //remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);
             cout<<"Running full GLOBAL Bundle Adjustment..."<<endl;
             Ceres_Bundler BA_ceres;
             new_k = BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_global_opt,Cloud,true,true,false,bUniqueCamera,0); //global BA
             next_ba = 30;

             if(new_k.rows > 0)
             {
	             K.at<double>(0,0) = new_k.at<double>(0,0);
	             K.at<double>(1,1) = new_k.at<double>(0,0);

	             D.at<double>(0,0) = new_k.at<double>(0,1);

	             //cout<<K<<endl; getchar();
	             //cout<<D<<endl; getchar();
         	 }

     	 }
         else if(ba_counter == next_ba) // GLOBAL BA
         {
         	cout<<"-----------------------------------------------"<<endl;
            remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
            correct_cloud(Cloud);
            //remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);
             cout<<"Running GLOBAL Bundle Adjustment..."<<endl;
             Ceres_Bundler BA_ceres;
             new_k = BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_global_opt,Cloud,true,true,false,bUniqueCamera,0);

             if(LOCAL_BA)
             next_ba = (float)next_ba * GBA_FACTOR;//2;
         	 else 
         	 next_ba += skip_ba;

         	if(new_k.rows > 0)
         	{

	             K.at<double>(0,0) = new_k.at<double>(0,0);
	             K.at<double>(1,1) = new_k.at<double>(0,0);

	             D.at<double>(0,0) = new_k.at<double>(0,1);


             cout<<K<<endl; //getchar();
         	}

     	 }
     	 else if(LOCAL_BA && next_ba/GBA_FACTOR > LOCAL_WSIZE && ba_counter%(LOCAL_WSIZE) == 0 && ba_counter >= LOCAL_WSIZE) //LOCAL BUNDLE ADJUSTMENT
     	 {
            remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
            correct_cloud(Cloud);
            //remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);
            cout<<"Running LBA with wsize = "<<LOCAL_WSIZE<<" ..."<<endl;
            Ceres_Bundler BA_ceres;
            BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_global_opt,Cloud,true,true,false,bUniqueCamera,LOCAL_WSIZE);

            //getchar();

     	 }




     	 if(ba_counter%32 ==0)
     	 {
            remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
            correct_cloud(Cloud);
     	 	//remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);
     	 	save_PLY(imgs_features,Cloud,gps_data,img_dir+"/result/");
     	 	cout<<"Saved PLY."<<endl;
     	 	//getchar();
     	 }





    }




 //////////////FINAL BUNDLE ADJUSTMENT
 cout<<"Final Bundle Adjustment"<<endl;

      //  remove_outliersfromcloud(Cloud);
       /*remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);*/
       remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
       correct_cloud(Cloud);
       //remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);
	    //remove_outliers_by_reprojerror_2(imgs_features,K,imgs_features[0]->cols,Cloud);
        ///////BUNDLE ADJUSTMENT////////

        if(GLOBAL_BA)
        {
        // adjust_local_ba_cvsba(Cloud, Kvec, Dvec, imgs_features,imgs_features[0]->rows,imgs_features[0]->cols,true,idx_local_opt,true);
         idx_local_opt.clear();

         for(int m=0; m < imgs_features.size();m++)
         {
            if(imgs_features[m]->triangulated)
            idx_local_opt.push_back(m);
        	else
        	cout<<">Missing camera: "<<imgs_name[m]<<endl<<endl;

        	if(false) // adjust all camera params in the end
        	imgs_features[m]->had_intrinsics = false;
		 }



          Ceres_Bundler BA_ceres; //Call twice to strongly minimize

          BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_local_opt,Cloud,true,true,false,bUniqueCamera,0); //global BA
          //BA_ceres.Adjust(imgs_features,Kvec,Dvec,idx_local_opt,Cloud,true,true,false,bUniqueCamera,0); //global BA
        }

       // remove_outliersfromcloud(Cloud);
        /*remove_outliers_carefully(imgs_features,K,imgs_features[0]->cols);*/
        //remove_outliers_by_reprojerror(imgs_features,K,imgs_features[0]->cols);
        //remove_outliers_by_reprojerror_2(imgs_features,K,imgs_features[0]->cols,Cloud);





 //save_pointcloud(Cloud);



 //if(show_final_result)
 //show_pointCloud(Cloud);

 /*for(int i=0;i<G.rows;i++)
  for(int j=i+1;j<G.cols;j++)
  if(G.at<double>(i,j)!=0)
    stereo_Rectify(imgs[i],imgs[j],imgs_features[i]->P_mat, imgs_features[j]->P_mat); // finds the disparity map for the baseline triang after ba
*/

//old_code.cpp

double sum_reprjerr=0;
double posic = 0;

/*for(int i=0;i<Cloud.size();i++)
if(Cloud[i]->is_in_cloud)
{

    sum_reprjerr+= Cloud[i]->reproj_error/(double)Cloud[i]->owners.size();
    posic++;
}

*/

   //cout<<"Sum reprj error: "<<sum_reprjerr<<endl;
   // cout<<"Mean reproj error:"<<sum_reprjerr/posic; //


    //reconstruct_surface(Cloud);
     SFM2PMVS (imgs_features, K, img_dir, imgs_name, imgs,output_path,down_factor);
     SFM2BUNDLER(imgs_features,K,img_dir,imgs_name,imgs,Cloud,output_path); //
     save_elapsed_time(output_path);
     save_PLY(imgs_features,Cloud,gps_data,img_dir+"/result/");

     Kvec.clear();
     Dvec.clear();
     K.release();
     D.release();
     distortion.release();
     p1_img.release();
     p2_img.release();

	return 0;


}


Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                   cv::Matx34d P,       //camera 1 matrix
                   Point3d u1,      //homogenous image point in 2nd camera
                   cv::Matx34d P1       //camera 2 matrix
                                   )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    Mat A = (Mat_<float>(4,3)<<u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
              );
   Mat B = (Mat_<float>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                      -(u.y*P(2,3)  -P(1,3)),
                      -(u1.x*P1(2,3)    -P1(0,3)),
                      -(u1.y*P1(2,3)    -P1(1,3)));

 Mat X; //= (Mat_<float>(4,1));
//cout<<"\n-------------------------------------\n";



    solve(A,B,X,DECOMP_SVD);
   //cout<<"\n"<<A<<"\n";
   //cout<<"\n"<<B<<"\n";
  // waitKey(0);
    return X;


}

void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w)
 {
	//Using OpenCV's SVD
	cv::SVD svd(E,cv::SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
}


bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2)
{

	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);
//
 Mat diag = (Mat_<double>(3,3) << 1,0,0,   0,1,0,   0,0,0);
 Mat_<double> newE = svd_u*diag*svd_vt;
 TakeSVDOfE(newE,svd_u,svd_vt,svd_w);
////
	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		      1,0,0,
	        	0,0,1);
	Matx33d Wt(0,1,0,
	        	-1,0,0,
	         	0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3

	return true;
}


Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
											Matx34d P,			//camera 1 matrix
											Point3d u1,			//homogenous image point in 2nd camera
											Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4,1);
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) {break;}

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
												  -(u.y*P(2,3)	-P(1,3))/wi,
												  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );

		solve(A,B,X_,DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}


