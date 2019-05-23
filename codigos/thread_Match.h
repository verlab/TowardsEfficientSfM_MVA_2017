// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>
#include <queue>
#include <set>

#define num_of_threads 20

using namespace std;
using namespace cv;





double match_parallel(imgMatches*im1, imgMatches *im2,Mat &F_mat, int qualDescritor, vector<Point2f> &cam_1, vector<Point2f> &cam_2, vector<long> &idx_cam1, vector<long> &idx_cam2,int &homog_inliers,bool use_flann, double inliers_threshold, int id_cam1, int id_cam2) //returns qtd de inliers
{
 //use_flann = false;
    //calculating matching
    //
    Mat img1,img2;

    int qualnorm;
    bool write_img= false;
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



    //////////////////////////////////////////////////


    //FlannBasedMatcher fnn_matcher2(new flann::LshIndexParams(20,10,2));
       vector<vector<DMatch> > k_matches;
       vector<vector<DMatch> > k_matches2;



	if(use_flann)
	{

     if(qualnorm == NORM_L2)
     {
  	     fnn_matcher.knnMatch(im2->descriptors_img, im1->descriptors_img,k_matches,2,Mat(),false);
         fnn_matcher.knnMatch(im1->descriptors_img, im2->descriptors_img,k_matches2,1,Mat(),false); //cross-validation
      }
  	     else
  	     ;///fnn_matcher2.match(im2->descriptors_img, im1->descriptors_img, matches);

  }
	else
	matcher.match(im2->descriptors_img, im1->descriptors_img, matches);
    //cout<<matches.size()<<endl;
    //desenha matches
    vector<DMatch> good_matches;

    double min_dist=9999, max_dist=0;

    if(use_flann)
    {
      //Criterion ratio test and cross_validation
      for(int i=0;i<k_matches.size();i++)
      { //cout<<k_matches2[i][0].queryIdx<<endl; getchar();
      	if(k_matches[i][0].distance < 0.80*k_matches[i][1].distance && k_matches2[k_matches[i][0].trainIdx][0].trainIdx==k_matches[i][0].queryIdx)
      		good_matches.push_back(k_matches[i][0]);
      }

    }
    else
        good_matches = matches;

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
    //cout<<"PATH "<<path<<endl;

	//imwrite( path, img_matches );

   	  /********Testing vector consistency******/


   	  if(false)
    {
        vector<bool> inconsist (im1->keypoints_img.size(),false);
        int inconsist_c = 0;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            int idx = good_matches[i].trainIdx;
            if(inconsist[idx])
                inconsist_c++;

           inconsist[idx] = true;
        }


        if(inconsist_c>15)
        {cout<<"-----Estourou inliers "<<inconsist_c<<endl;}


    }


    vector<Point2f> cam1;
    vector<Point2f> cam2;

    vector<Point2f> n_cam1;
    vector<Point2f> n_cam2;
  //  Mat points3D;
    std::vector<uchar> inliers(cam1.size(),0);




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

    if(im1->rows < im1->cols)
      threshold = im1->cols;
    else
      threshold = im1->rows;

    if(im2->rows > threshold)
      threshold = im2->rows;

    if(im2->cols > threshold)
      threshold = im2->cols;

    Mat fmat2;
 //   fmat2 = cv::findEssentialMat(cam1, cam2, Kvec[id_cam1].at<double>(0,0),Point2d(Kvec[id_cam1].at<double>(0,2),Kvec[id_cam1].at<double>(1,2)),cv::RANSAC, 0.999,(0.00220*(double)im1->cols),inliers);

  if(cam1.size()>7 && cam2.size()>7 && cam1.size() == cam2.size()) //0.00220
   fmat2 = findFundamentalMat(cam1, cam2, FM_RANSAC, 0.006*threshold, 0.99,inliers); //threshold 7.5 pxs for 1280x720 images
  else
  {
    cam1.clear(); cam2.clear();
    F_mat = (Mat_<double>(3,3)<<0,0,0,0,0,0,0,0,0);
    return 0;
  }


     // Mat fmat2 = openMVG::sfm::FindFundamentalMat_AC_RANSAC(cam1, cam2,Kvec[id_cam1], Kvec[id_cam2], im1->cols, im1->rows, im2->cols, im2->rows, threshold, inliers);
    //cout<<"Found AC threshold: "<<threshold<<endl<<endl;



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

    if(write_img)
    {cv::resize(img_matches,img_matches,Size(),0.5,0.5);
    imwrite(path2, img_matches);

    }



cam1.clear();
cam2.clear();
homog_inliers = 0;


   if(good_matches_RANSAC.size() > inliers_threshold)
   {
      for( int i = 0; i < good_matches_RANSAC.size(); i++ )
      {
      cam1.push_back( im1->keypoints_img[ good_matches_RANSAC[i].trainIdx ].pt );
  	  cam2.push_back( im2->keypoints_img[ good_matches_RANSAC[i].queryIdx ].pt );

      idx_cam1.push_back(good_matches_RANSAC[i].trainIdx);
      idx_cam2.push_back(good_matches_RANSAC[i].queryIdx);

   	  }



      /*********************Homography computation using inliers******************/

     std::vector<uchar> inliers_homography(cam1.size(),0);

     Mat mat_homog = findHomography(cam1, cam2, FM_RANSAC,(0.006*(double)im1->cols),inliers_homography);



      std::vector<uchar>::const_iterator itIn_h= inliers_homography.begin();


      // number of inliers homography
      for ( ;itIn_h!= inliers_homography.end(); ++itIn_h)
      {
          if (*itIn_h) //  if inlier
          { // it is a valid match
             homog_inliers++;

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



void match_batch(vector<imgMatches*> imgs_features, vector<pairMatch*> *vec_match, vector<pair<int,int> > batch, double inliers_threshold) //receiving a batch of pairs to match
{
    int progress=0;


        for(int k=0;k<batch.size();k++)
        {
              int i=batch[k].first;
              int j=batch[k].second;

             vector<Point2f> cam_1;
             vector<Point2f> cam_2;
             vector<long> idx_cam_1, idx_cam_2;
             int w_homog_inliers=0;

             Mat F_mat = (Mat_<double>(3,3)<<0,0,0,0,0,0,0,0,0);
             pairMatch *pm = new pairMatch;

            if(false)
           	;// G.at<double>(i,j) = (double)match_pair(imgs_features[i], imgs_features[j],F_mat,qualDescritor, cam_1, cam_2,idx_cam_1,idx_cam_2, Cloud,imgs[i],imgs[j],w_homog_inliers,use_flann,good_matches_percentile,inliers_threshold);
            else if(imgs_features[i]->keypoints_img.size()> 10 && imgs_features[j]->keypoints_img.size() > 10)
            {

            cv::Mat Maux;
            Maux.create(1,1,CV_64F);

            match_parallel(imgs_features[i],imgs_features[j],F_mat,0, cam_1, cam_2,idx_cam_1,idx_cam_2,w_homog_inliers,true,inliers_threshold,i,j);

            }
            pm->cam1 = cam_1;
            pm->cam2 = cam_2;
            pm->idx_cam1 = idx_cam_1;
            pm->idx_cam2 = idx_cam_2;


            pm->F_mat = F_mat;



            pm->triangulated = false;
            pm->h_tested = false;
            pm->homog_inliers = w_homog_inliers;

            //pairs->at(i)[j] = pm;
            vec_match->push_back(pm);

            progress++;

            if(progress%10==0)
            cout<<"done pair ("<<i<<","<<j<<") progress:"<<100*(progress/float(batch.size()))<<"%"<<endl;


        }



}




/*
bool check_connectivity(vector<pKeypoint*> owners, pKeypoint* pk)
{

  bool connected = true;


  for(int i=0;i< owners.size(); i++)
  {
      bool found = false;

      pair<int,int> query = make_pair(owners[i]->img_owner->id, owners[i]->master_idx);

      for(int j=0;j<pk->matches.size();j++)
      {
        if(pk->matches[j].first == query.first && pk->matches[j].second == query.second)
        {
          found = true; break;
        }
      }

      if(!found)
      connected = false;

  }

  return connected;
}
*/

bool check_connectivity(vector<pKeypoint*> &owners, pKeypoint* pk)
{

  int connected = 0;
  int threshold = std::max((int)2,(int)(owners.size()/5));

  for(int i=0;i< owners.size(); i++)
  {
      bool found = false;

      pair<int,int> query = make_pair(owners[i]->img_owner->id, owners[i]->master_idx);

      for(int j=0;j<pk->matches.size();j++)
      {
        if(pk->matches[j].first == query.first && pk->matches[j].second == query.second)
        {
          found = true; break;
        }
      }

      if(found)
      connected++;

  }

  if(connected < threshold)
    return false;
  
  return true;

  /*Enforce consistency*/

  for(int i=0;i< owners.size(); i++)
  {
    pKeypoint* pkp= owners[i];
    threshold = std::max((int)2,(int)(owners.size()/5));
    connected = 0;


    for(int j=0;j<owners.size();j++)
    {
       bool found = false;
       pair<int,int> query = make_pair(owners[j]->img_owner->id, owners[j]->master_idx);

       for(int k=0;k<pkp->matches.size();k++)
       {
          if(pkp->matches[k].first == query.first && pkp->matches[k].second == query.second)
           { found = true; break;}

       }

       if(found)
          connected++;
    }

    if(connected < threshold) //incosistent -- remove this pkeypoint from track
    {

      cloudPoint * cld = pkp->cloud_p;

      int m;

      for(m=0;m<cld->owners.size();m++)
        if(cld->owners[m]==pkp)
          break;


       if(m < cld->owners.size()) //found
       {
        cld->owners.erase(cld->owners.begin()+m);
        i--;
       }

       if(cld->owners.size() < 2)
       {
          for(m=0;m<cld->owners.size();m++)
            cld->owners[m]->cloud_p = NULL;

          delete cld;

          return false;
       }

       pkp->cloud_p = NULL;

    }
  }

  return true;
}


void build_projections_BFSearch(Mat &G, vector<vector<pairMatch*> > pairs,vector<imgMatches*> imgs,  vector<pair<int,int> > all_pairs, int inliers_threshold, vector<cloudPoint*> &Cloud) //perform a breadth first search to merge tracks into a 3D point (pcloud)
{
  //FIRST STAGE ---- COMPUTE ADJACENCY LIST -----
 for(int k=0;k<all_pairs.size();k++)
 {
    if(pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size() > inliers_threshold)
   {
        imgMatches* im1 = imgs[all_pairs[k].first];
        imgMatches* im2 = imgs[all_pairs[k].second];

      for( int i = 0; i < pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1.size(); i++ )
      {

        int idx1 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1[i]; //idx of pkeypoint on image 1
        int idx2 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam2[i];

        im1->pkeypoints[idx1]->matches.push_back(make_pair(all_pairs[k].second,idx2));
        im2->pkeypoints[idx2]->matches.push_back(make_pair(all_pairs[k].first,idx1));

      }

   }
  }


    
  for(int i=0;i<imgs.size();i++) //for each image, compute tracks
  {
    for(int j=0;j<imgs[i]->pkeypoints.size();j++) //for each pkeypoint
    {
      if(!imgs[i]->pkeypoints[j]->checked && imgs[i]->pkeypoints[j]->matches.size() > 0) //if it has any match at all, perform a BF search
      {
        queue<pKeypoint*> recursion;
        vector<pKeypoint*> component;
        set<int> cams_visited;


        //component.push_back(imgs[i]->pkeypoints[j]);
        recursion.push(imgs[i]->pkeypoints[j]);
        imgs[i]->pkeypoints[j]->checked = true;
        cams_visited.insert(imgs[i]->pkeypoints[j]->img_owner->id);

        while(!recursion.empty()) //find the connected component
        {
          pKeypoint* current = recursion.front(); recursion.pop();

            component.push_back(current);

            for(int k=0; k< current->matches.size(); k++) // for all its children
            {
              pKeypoint *child = imgs[current->matches[k].first]->pkeypoints[current->matches[k].second];

                if(!child->checked && cams_visited.find(child->img_owner->id) == cams_visited.end())//if not checked and the camera is not in the visited set
                 { 
                   recursion.push(child); 
                   cams_visited.insert(child->img_owner->id);
                   child->checked = true;
                 }
                        
            }
          
        }
          if(false)//component.size()>150)
          {

             /*for (std::set<int>::iterator it=cams_visited.begin(); it!=cams_visited.end(); ++it)
                   std::cout << ' ' << *it<<endl;
              */
                cout<<"cams_visited size ="<<cams_visited.size()<<endl;
               
            for(int i=0;i<component.size();i++)
                cout<<component[i]->img_owner->id<<endl;

              getchar();
          }
          


          //create a new 3D point for this track of points
          cloudPoint *cp = new cloudPoint;
          cp->is_in_cloud = false;
          cp->idx = -1;
          cp->x = cp->y = cp->z = 0;
          cp->reproj_error = 0;
          cp->can_ba = 0;
          cp->added = false;

          for(int k=0;k< component.size(); k++)
          {
            component[k]->cloud_p = cp;
            cp->owners.push_back(component[k]);
          }

          //Cloud.push_back(cp);
          //cout<<"Track of size: "<<component.size()<< endl; getchar();

      } 
    }

  }

}


void build_projections(Mat &G, vector<vector<pairMatch*> > pairs,vector<imgMatches*> imgs,  vector<pair<int,int> > all_pairs, int inliers_threshold, vector<cloudPoint*> &Cloud)
{


 //FIRST STAGE ---- COMPUTE MATCHES VECTOR -----
 for(int k=0;k<all_pairs.size();k++)
 {
    if(pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size() > inliers_threshold)
   {
        imgMatches* im1 = imgs[all_pairs[k].first];
        imgMatches* im2 = imgs[all_pairs[k].second];

      for( int i = 0; i < pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1.size(); i++ )
      {

        int idx1 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1[i]; //idx of pkeypoint on image 1
        int idx2 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam2[i];

        im1->pkeypoints[idx1]->matches.push_back(make_pair(all_pairs[k].second,idx2));
        im2->pkeypoints[idx2]->matches.push_back(make_pair(all_pairs[k].first,idx1));

      }
   }
  }





for(int k=0;k<all_pairs.size();k++)
{
      if(pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size() > inliers_threshold) // dont create a point cloud for outlier pair, cuz it doesn't makes sense.
     {//G.at<double>(all_pairs[k].first,all_pairs[k].second) = pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size();
      for( int i = 0; i < pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size(); i++ )
      {



      int idx_c1,idx_c2;

      imgMatches* im1 = imgs[all_pairs[k].first];
      imgMatches* im2 = imgs[all_pairs[k].second];

      idx_c1 = im1->pkeypoints[pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1[i]]->master_idx;
      idx_c2 = im2->pkeypoints[pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam2[i]]->master_idx;

      //adding corresponding points into a common point in cloud
      if(im1->pkeypoints[idx_c1]->cloud_p ==NULL && im2->pkeypoints[idx_c2]->cloud_p ==NULL ) //not set yet
      {
          cloudPoint *cp = new cloudPoint;
          cp->is_in_cloud = false;
          cp->idx = -1;
          cp->x = cp->y = cp->z = 0;
          cp->reproj_error = 0;
          cp->can_ba = 0;
          cp->added = false;


          cp->owners.push_back(im1->pkeypoints[idx_c1]);
          cp->owners.push_back(im2->pkeypoints[idx_c2]);

          im1->pkeypoints[idx_c1]->cloud_p = cp;
          im2->pkeypoints[idx_c2]->cloud_p = cp;


         // Cloud.push_back(cp);

      }
      else if(im1->pkeypoints[idx_c1]->cloud_p ==NULL && im2->pkeypoints[idx_c2]->cloud_p != NULL )
      {
           if( check_connectivity(im2->pkeypoints[idx_c2]->cloud_p->owners, im1->pkeypoints[idx_c1]) )
           {
             im1->pkeypoints[idx_c1]->cloud_p = im2->pkeypoints[idx_c2]->cloud_p;
             im2->pkeypoints[idx_c2]->cloud_p->owners.push_back(im1->pkeypoints[idx_c1]);
           }
      }
      else if(im1->pkeypoints[idx_c1]->cloud_p != NULL && im2->pkeypoints[idx_c2]->cloud_p == NULL )
      {
            if( check_connectivity(im1->pkeypoints[idx_c1]->cloud_p->owners, im2->pkeypoints[idx_c2]))
            {
              im2->pkeypoints[idx_c2]->cloud_p = im1->pkeypoints[idx_c1]->cloud_p;
              im1->pkeypoints[idx_c1]->cloud_p->owners.push_back(im2->pkeypoints[idx_c2]);
            }
      }
      else if( im2->pkeypoints[idx_c2]->cloud_p != im1->pkeypoints[idx_c1]->cloud_p)
      { //merge 2 with 1 (pass elements from 2 to the largest track 1)

          cloudPoint *c1, *c2;

          if(im1->pkeypoints[idx_c1]->cloud_p->owners.size() > im2->pkeypoints[idx_c2]->cloud_p->owners.size())
          {
            c1 = im1->pkeypoints[idx_c1]->cloud_p;
            c2 = im2->pkeypoints[idx_c2]->cloud_p;
          }
          else
          {
            c2 = im1->pkeypoints[idx_c1]->cloud_p;
            c1 = im2->pkeypoints[idx_c2]->cloud_p;
          }

          for(int j=0; j < c2->owners.size();j++)
          {
            

             if( check_connectivity(c1->owners, c2->owners[j]))
              {
              c1->owners.push_back(c2->owners[j]);
              c2->owners[j]->cloud_p = c1;
              c2->owners.erase(c2->owners.begin()+j); j--;
              }
            
           
          /* bool could = false;
           for(int m=0; m<c2->owners.size();m++)
            { 
              if( check_connectivity(c1->owners, c2->owners[m]))
              {
              c1->owners.push_back(c2->owners[m]);
              c2->owners[m]->cloud_p = c1;
              c2->owners.erase(c2->owners.begin()+m); m--;

              could = true;
              }

            }

            if(!could)
              break;
            */ 
            
          }

          if(c2->owners.size() < 2) //eliminate track and point3d
          {
            for(int j=0; j<c2->owners.size();j++)
              c2->owners[j]->cloud_p = NULL;

            delete c2;
          }


         }

      }
  }
  else
  G.at<double>(all_pairs[k].first,all_pairs[k].second) = 0;


  }

}

bool sort_tracks(pair<int, int> pkp1, pair<int,int> pkp2)
{
    return pkp1.second > pkp2.second;
}

/*void build_projections(Mat &G, vector<vector<pairMatch*> > pairs,vector<imgMatches*> imgs,  vector<pair<int,int> > all_pairs, int inliers_threshold, vector<cloudPoint*> &Cloud)
{


 //FIRST STAGE ---- COMPUTE TRACKS -----
 for(int k=0;k<all_pairs.size();k++)
 {
    if(pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size() > inliers_threshold)
   {
      for( int i = 0; i < pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1.size(); i++ )
      {
        imgMatches* im1 = imgs[all_pairs[k].first];
        imgMatches* im2 = imgs[all_pairs[k].second];

        int idx1 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam1[i]; //idx of pkeypoint on image 1
        int idx2 =  pairs[all_pairs[k].first][all_pairs[k].second]->idx_cam2[i];

        im1->pkeypoints[idx1]->matches.push_back(make_pair(all_pairs[k].second,idx2));
        im2->pkeypoints[idx2]->matches.push_back(make_pair(all_pairs[k].first,idx1));

      }
   }
   else
    G.at<double>(all_pairs[k].first,all_pairs[k].second) = 0;
  }



  //SECOND STAGE ----- REGISTER POINTCLOUD TRACKS -----

    for(int i=0;i<imgs.size();i++)
    {
      vector<pair<int,int> > sorted_pk;

      for(int j=0;j<imgs[i]->pkeypoints.size();j++)
        sorted_pk.push_back(make_pair(j,imgs[i]->pkeypoints[j]->matches.size()));

       std::sort(sorted_pk.begin(),sorted_pk.end(),sort_tracks);

       for(int j=0;j<sorted_pk.size();j++)
        {

          if(imgs[i]->pkeypoints[sorted_pk[j].first]->cloud_p == NULL && imgs[i]->pkeypoints[sorted_pk[j].first]->matches.size() > 0)
          {
            cloudPoint *cp = new cloudPoint;
            cp->is_in_cloud = false;
            cp->idx = -1;
            cp->x = cp->y = cp->z = 0;
            cp->reproj_error = 0;
            cp->can_ba = 0;
            cp->added = false;

            imgs[i]->pkeypoints[sorted_pk[j].first]->cloud_p = cp;
            cp->owners.push_back(imgs[i]->pkeypoints[sorted_pk[j].first]);

            for(int l=0;l < imgs[i]->pkeypoints[sorted_pk[j].first]->matches.size();l++)
            {
              pair<int,int> _match = imgs[i]->pkeypoints[sorted_pk[j].first]->matches[l];
              pKeypoint * pk_2 = imgs[_match.first]->pkeypoints[_match.second];

              if(pk_2->cloud_p == NULL)
              {
                pk_2->cloud_p = cp;
                cp->owners.push_back(pk_2);
              }
              else //check the number of matches and choose the best
              {
                int c_match1 = 0, c_match2 = 0;

                cloudPoint *cld = pk_2->cloud_p;
                
                //for(int m=0; m < cld->owners.size();)
              }

            }
              if(cp->owners.size() < 2)
              {
                imgs[i]->pkeypoints[sorted_pk[j].first]->cloud_p = NULL;
                delete cp;
              }
          }

        }

    }





}
*/

void update_slave_pkeypoints(vector<imgMatches*> imgs)
{

    for(int i=0;i<imgs.size();i++)
    {
        cloudPoint* cp = NULL;
        int master = -1;

        for(int j=0;j<imgs[i]->pkeypoints.size();j++)
        {
            if(imgs[i]->pkeypoints[j]->cloud_p!=NULL)
            {
                cp = imgs[i]->pkeypoints[j]->cloud_p;
                master = imgs[i]->pkeypoints[j]->master_idx;


            }

            if(imgs[i]->pkeypoints[j]->cloud_p==NULL && imgs[i]->pkeypoints[j]->master_idx == master)
                imgs[i]->pkeypoints[j]->cloud_p = cp;
        }
    }
}

bool sort_pt3d(cloudPoint *c1, cloudPoint* c2)
{
  return c1->owners.size() > c2->owners.size();
}

void fill_pt3d_vector(vector<imgMatches*> imgs, vector<cloudPoint*> &Cloud)
{
    for(int i=0;i<imgs.size();i++)
    {
        for(int j=0;j<imgs[i]->pkeypoints.size();j++)
        {
           cloudPoint* cp = imgs[i]->pkeypoints[j]->cloud_p;

           if(cp != NULL && !cp->added)
           {
                Cloud.push_back(cp);
                cp->added = true;
           }
        }
    }

    std::sort(Cloud.begin(), Cloud.end(), sort_pt3d);

    for(int i=0;i<25;i++)
    {
      cout<<"Track of size "<<Cloud[i]->owners.size()<<endl;

     /* for(int j=0;j<Cloud[i]->owners.size();j++)
      { int count = 0;
        for(int k=0;k<Cloud[i]->owners.size();k++)
          if(Cloud[i]->owners[j] == Cloud[i]->owners[k])
            count++;

          cout<<"Number of repetitions = "<<count<<endl; getchar();
      }
     */
    }

    long c_points =0;
    for(int i=0;i<Cloud.size();i++)
    {
        if(Cloud[i]->owners.size() > 2)
          c_points++;
    }

    cout<<"Cloud initial size: "<<Cloud.size()<<endl;
    cout<<"Stable points: "<<c_points<<endl;
    //getchar();
}

int match_pair_parallel(Mat &G, vector<vector<pairMatch*> > &pairs,vector<imgMatches*> imgs, vector<cloudPoint*> &Cloud, double inliers_threshold)
{
	    vector<thread> mythreads;
	    vector<pair<int,int> > all_pairs;
	    vector<vector<pair<int, int> > > batches;
	    vector<vector<pairMatch*>*> vec_matches;



    for(int i=0;i<G.rows;i++)
         for(int j=i+1; j<G.cols; j++)
         {
             if(G.at<double>(i,j) == 1)
             all_pairs.push_back(make_pair(i,j));
         }


    int parts =(all_pairs.size()/(float)num_of_threads) +1;
    int pos=0;

         for(int i=0;i<num_of_threads;i++)
         {

            batches.push_back(vector<pair<int,int> >());
            vec_matches.push_back(new vector<pairMatch*>());

              for(int j=0;j<parts && pos < all_pairs.size(); j++)
              {
                batches[i].push_back(all_pairs[pos++]);
              }

          }

	   for(int i=0;i<num_of_threads;i++)
	{
	    mythreads.push_back(thread(match_batch,imgs,vec_matches[i],batches[i],inliers_threshold));

	}

	 cout<<"waiting for all to finish... total pairs = "<<pair_count<<endl;


	   for(int i=0;i<num_of_threads;i++)
	{
	     mythreads[i].join();
	}
  long check_pairs=0;
	for(int i=0;i<num_of_threads;i++)
	{
        for(int j=0;j<batches[i].size();j++)
        {
            pairs[batches[i][j].first][batches[i][j].second] = vec_matches[i]->at(j); //check_pairs++;

        }
	}

  /*if(check_pairs==all_pairs.size())
    {cout<<"PAIRS CHECKED OK"<<endl; getchar();}
  else
    {cout<<"PAIRS ARE WRONG"<<endl; getchar();}
  */

  //update_slave_pkeypoints(imgs);
   for(int k=0;k<all_pairs.size();k++)
   {
     if(pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size() > inliers_threshold) // dont create a point cloud for outlier pair, cuz it doesn't make sense.
          G.at<double>(all_pairs[k].first,all_pairs[k].second) = pairs[all_pairs[k].first][all_pairs[k].second]->cam1.size();
   }

    cout<<"total pairs:"<<pair_count<<endl;//getchar();
	    return 0;
}
