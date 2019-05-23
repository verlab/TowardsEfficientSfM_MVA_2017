// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <utility>

#define MAX_N 3000
#define MAX_T 600
#define MAX_Q 40

#include <string>



using namespace std;
using namespace cv;

/*struct imgMatches
{
	Mat descriptors_img;
	vector<KeyPoint> keypoints_img;
	Mat image;
	string img_path;
};
*/


struct VocabTree
{

	vector<VocabTree*> nodes;
	Mat centroid;
	vector<int> img_idx;

};

vector<imgMatches*> images;






void build_tree(vector<pair<int,int> > desc_idx, VocabTree* node)
{

	int K_factor = 9;
	cv::Mat descriptors;
	cv::Mat labels;
	cv::Mat centers;
	vector< vector < pair <int,int> > > branches;


	if(desc_idx.size() > K_factor*2)
	{
		for(int i=0;i<K_factor;i++)
			branches.push_back(vector<pair<int,int> >());

		for(long i=0;i<desc_idx.size();i++)
			descriptors.push_back(images[desc_idx[i].first]->descriptors_img.row(desc_idx[i].second));


	        kmeans(descriptors, K_factor, labels,
	               TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
	               3, KMEANS_PP_CENTERS, centers);


	        for(long i = 0; i < labels.rows; i++ )
	        {
	            int clusterIdx = labels.at<int>(i);
	            pair<int,int> p = desc_idx[i];

	            branches[clusterIdx].push_back(p);
	        }

	        for(int i=0; i<centers.rows;i++)
	        {
	        	 VocabTree *n_node = new VocabTree;
	        	 node->nodes.push_back(n_node);

	        	 n_node->centroid = centers.row(i);

	        }

	        for(int i=0;i<node->nodes.size(); i++)
	        	build_tree(branches[i],node->nodes[i]);

    }

}

void get_descs_idx(vector<pair<int,int> > &descs)
{
	for(int i=0;i<images.size();i++)
		for(int j=0;j<images[i]->keypoints_img.size() && j < MAX_T;j++)
			descs.push_back(make_pair(i,rand()%images[i]->keypoints_img.size()));

}

void insert_feature(VocabTree* tree, Mat ft,int id)
{

	VocabTree* aux = tree;
	double min_dist=99999;
	int min_idx;

	while(aux->nodes.size()>0) //isnt leaf
	{
		for(int i=0;i<aux->nodes.size();i++)
		{
			double dist = cv::norm(ft-aux->nodes[i]->centroid);
			if(dist<min_dist)
			{
				min_dist = dist; min_idx = i;
			}
		}

		aux = aux->nodes[min_idx];
	}

	aux->img_idx.push_back(id);
}

void train_image(VocabTree *tree, imgMatches* img, int id)
{
	for(int i=0;i< img->descriptors_img.rows && i < MAX_N;i++)
		insert_feature(tree, img->descriptors_img.row(i), id);
}

void build_database(VocabTree *tree)
{
	for(int i=0;i<images.size();i++)
		train_image(tree,images[i],i);
}

bool sort_highscores(pair<int,float> p1, pair<int,float> p2)
{
	return p1.second > p2.second;
}

vector<pair<int, float> > query_img(VocabTree* tree, imgMatches* img)
{

	vector<float> hist(images.size(),0);
	vector<pair<int, float> > highscores;

	for(int i=0;i< img->descriptors_img.rows && i < MAX_N;i++)
	{

		VocabTree* aux = tree;
        int depth=0;

		while(aux->nodes.size()>0) //isnt leaf
		{
			double min_dist=99999;
			int min_idx = 0;

			for(int j=0;j<aux->nodes.size();j++)
			{
				double dist = cv::norm(img->descriptors_img.row(i)-aux->nodes[j]->centroid);
				if(dist<min_dist)
				{
					min_dist = dist; min_idx = j;
				}
			}

			aux = aux->nodes[min_idx]; depth++;
		}

		for(int k=0;k<aux->img_idx.size();k++)
			hist[aux->img_idx[k]]+=100.0/(float)(depth*depth);

	}


	for(int i=0;i<hist.size();i++)
		highscores.push_back(make_pair(i,hist[i])); //index, score

	std::sort(highscores.begin(),highscores.end(),sort_highscores);

	return highscores;

}

vector<vector<int> >vocabulary_tree_prune(vector<imgMatches*> imgs)
{
   vector<vector<int> > full_query;

    images = imgs;

    vector<pair<int,int> > all_descs;
	get_descs_idx(all_descs);

	VocabTree *tree = new VocabTree;
	cout<<"starting vocab tree construction..."<<endl;
	build_tree(all_descs, tree);
	cout<<"tree has been built."<<endl;
	build_database(tree);
	cout<<"tree is now indexed with database."<<endl;


    for(int i=0;i<images.size();i++)
    {
	vector<pair<int, float> > bests = query_img(tree,images[i]);

	//cout<<"querying done"<<endl;
	vector<int> q_idx;

	for(int j=0;j<bests.size() && j< MAX_Q + 1;j++)
        q_idx.push_back(bests[j].first);
		//cout<<images[bests[j].first]->img_path<<" - " <<bests[j].second<<endl;

		full_query.push_back(q_idx);

    }

	cout<<"Done"<<endl;

	return full_query;
}

/*int main()
{

	char * path = new char[256];
	sprintf(path,"/home/vnc/test_datasets/Teste_Mina");

	vector<imgMatches*> _imgs;

	open_imgs_dir(path, _imgs, 0.3);

	for(int i=0;i<_imgs.size();i++)
	{find_keypoints(_imgs[i]); cout<<i<<endl;}

	cout<<"Matching pairs now........."<<endl;

	vocabulary_tree_prune(_imgs);

	cout<<"full query performed."<<endl;

	return 0;
}
*/
