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


using namespace std;
using namespace cv;


void read_keyfile(string img, vector<KeyPoint> &keypoints, Mat &desc) //read LOWE'S .key format file containing the keypoints
{
	FILE *f = fopen( img.c_str(), "r");
	int n,d,bin;
	float x,y,s,o;

	fscanf(f,"%d %d",&n,&d);

	for(int i=0;i<n;i++)
	{

		fscanf(f,"%f %f %f %f",&y,&x,&s,&o);

		KeyPoint novo;
		novo.pt.x = x;
		novo.pt.y = y;
		novo.angle = o;
		novo.size = s;
		keypoints.push_back(novo);

		cv::Mat vec(1,128,CV_32F);

		for(int j=0; j<128;j++)
		{
			fscanf(f,"%d",&bin);
			vec.at<float>(0,j) = bin;
		}

		desc.push_back(vec);
	}


}

void rootSIFT(cv::Mat &descr)
{
    cv::normalize(descr,descr,1.0,cv::NORM_L1);
    cv::sqrt(descr,descr);
    //cv::normalize(descr,descr,1.0,cv::NORM_L2);
}

void compute_parallel_keypoints(imgMatches *im, Mat imgCena, int qualDescritor, string name)
{


    vector<KeyPoint> keypointsCena;
	Mat descriptorsCena;

	double major = (imgCena.rows>imgCena.cols)? imgCena.rows: imgCena.cols;
	double max_threshold = pow(major,1.1)+7000;

	if (qualDescritor ==0)
	{   //keypoints SIFT


	   // Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(19000, 3, 0.02, 5, 1.6);
		 Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(/*pow(major,1.1)+11000*/14000, 3, 0.02, 5, 1.6);
	   // SIFT* detector = new SIFT::SIFT(pow(imgCena.rows,1.3)*3.85, 3, 0.04, 5, 1.3); //0.0065   1.6*0.2  1.2*4.55  1.2*1.55  -- 4.55 sigma 1.7 //6.55 1.85
	    detector->detect(imgCena, keypointsCena);
	    cout<<"SIFT Detected: " << keypointsCena.size() <<" keypoints..."<<endl; ////





  	sort_keypoints_by_response(keypointsCena);

	  	if(keypointsCena.size() > (int)max_threshold)
	  	{
	  		keypointsCena.erase(keypointsCena.begin()+(int)max_threshold, keypointsCena.end());
	  		cout<<"but resized to ----> "<< (int)max_threshold<<endl;

	  	}

	}
  else if(qualDescritor == 1) //AKAZE
  {

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detect(imgCena, keypointsCena);
    //akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

  }
  else if(qualDescritor ==2) //READ precomputed features from .key file
  {
  	read_keyfile(name, keypointsCena, descriptorsCena);
  }



	if (qualDescritor==0)
	{ 	// SIFT
	    Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
		extractor->compute(imgCena, keypointsCena, descriptorsCena);

	}
  else if(qualDescritor==1)
  {
     Ptr<AKAZE> akaze = AKAZE::create();
    akaze->compute(imgCena, keypointsCena, descriptorsCena);

  }
    if(qualDescritor==0 || qualDescritor ==2)
        rootSIFT(descriptorsCena);


	im->keypoints_img = keypointsCena;
	im->descriptors_img = descriptorsCena;
	im->triangulated = false;
	im->already_rmv= false;
	im->rows = imgCena.rows;
	im->cols = imgCena.cols;
	im->had_intrinsics = false;
	im->EXIF = false;
	im->estimated = false;


	int i=0,j;
	/*
        while(i < keypointsCena.size())
    {

	    j = i;
        do
        {
            pKeypoint * pk = new pKeypoint;
            pk->cloud_p = NULL;
            pk->pt= im->keypoints_img[i].pt;
            pk->R = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[2];
            pk->G = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[1];
            pk->B = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[0];
            pk->master_idx = j;

            pk->img_owner = im;

            im->pkeypoints.push_back(pk);
            i++;

	    } while(i < keypointsCena.size() && keypointsCena[i-1].size == keypointsCena[i].size );
	}
	*/

	/*for(int i=0;i<keypointsCena.size();i++)
	{
		cout<<"Scale = "<<keypointsCena[i].size<<endl;
		cout<<"true idx = "<<i<<endl;
		cout<<"master idx = "<<im->pkeypoints[i]->master_idx<<endl<<endl;
		getchar();
	}
	*/

	for(int i=0;i<keypointsCena.size();i++)
	{

			pKeypoint * pk = new pKeypoint;
            pk->cloud_p = NULL;
            pk->pt= im->keypoints_img[i].pt;
            pk->R = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[2];
            pk->G = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[1];
            pk->B = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[0];
            pk->master_idx = i;

            pk->img_owner = im;

            im->pkeypoints.push_back(pk);

	}

/*	cout<<keypointsCena.size()<<endl;
	cout<<im->pkeypoints.size()<<endl;

	for(int i=0;i<keypointsCena.size();i++)
	{
        cout<<"Scale = "<<keypointsCena[i].size<<endl;
        cout<<"master idx = "<<im->pkeypoints[i]->master_idx<<endl;
        getchar();
	}
   // imgCena.release();

*/
}


void compute_batch(vector<imgMatches*> * vec, vector<Mat> imgs, vector<int> imgs_idx, string path, vector<string> imgs_name)
{

	for(int i=0;i<imgs_idx.size();i++)
	{
		string aux = imgs_name[imgs_idx[i]];
		string aux2;

		for(int j=0;aux[j] != '.'; j++)
		{
			aux2.push_back(aux[j]);
		}

		imgMatches* nova = new imgMatches;
		string name = path + "/" + aux2 + ".key"; //cout<<name<<endl;

		compute_parallel_keypoints(nova, imgs[imgs_idx[i]],0,name); //2 - load lowe .key files

		nova->id = imgs_idx[i];

		vec->push_back(nova);

		cout<<"progress: "<<(i/(float)imgs_idx.size())*100<<endl;

	}



}


void thread_features(vector<imgMatches*> &imgs_features, vector<Mat> imgs, string path, vector<string> imgs_name)
{

	 vector<thread> mythreads;
	 vector<vector<imgMatches*>* > computed_imgs;
	 vector<vector<int> > imgs_idx;


	 int d = imgs.size()/(float)num_of_threads;
	 int r = imgs.size()%num_of_threads;
	 int pos = 0;
	 int p_one;

   for(int i=0;i<num_of_threads;i++)
	{
		computed_imgs.push_back(new vector<imgMatches*>());
		imgs_idx.push_back(vector<int>());
		if(r>0)
		{
			p_one=1;
			r--;
		}
		else
			p_one=0;

		for(int j=0;j<d+p_one;j++)
		{
			imgs_idx[i].push_back(pos++);
		}

	    mythreads.push_back(thread(compute_batch,computed_imgs[i],imgs,imgs_idx[i],path,imgs_name));

	}


	  for(int i=0;i<num_of_threads;i++)
	{
	     mythreads[i].join();
	}

	for(int i=0;i<computed_imgs.size();i++)
	{
		for(int j=0;j<computed_imgs[i]->size();j++)
			imgs_features.push_back(computed_imgs[i]->at(j));

	}

}
