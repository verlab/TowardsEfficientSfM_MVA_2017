// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


void load_imgMatches(vector<imgMatches*> &imgs_features, string path, vector<string> imgs_name)
{

	int rows, cols;
	float x,y;
	unsigned char R, G, B;

	for(int i=0;i<imgs_name.size();i++)
	{
		FILE *f = fopen( (path+"/"+imgs_name[i]+".mat").c_str(), "rb");
		int j = 0;


		fread(&rows,sizeof(int),1,f); //
		fread(&cols,sizeof(int),1,f); //

		imgMatches* im = new imgMatches;

		im->triangulated = false;
		im->already_rmv= false;
		im->rows = rows;
		im->cols = cols;
		im->had_intrinsics = false;
		im->EXIF = false;
		im->estimated = false;
		im->id = i;

		fread(&x,sizeof(float),1,f); //x
		while(!feof(f))
		{

			pKeypoint *pk = new pKeypoint;

			
			fread(&y,sizeof(float),1,f); //y

			fread(&R,sizeof(unsigned char),1,f); //R
			fread(&G,sizeof(unsigned char),1,f); //G
			fread(&B,sizeof(unsigned char),1,f); //B

			pk->pt.x = x;
			pk->pt.y = y;

			pk->R = R;
			pk->G = G;
			pk->B = B;

			pk->cloud_p = NULL;
			pk->active = true;
			pk->checked= false;
			pk->master_idx = j++;
			pk->img_owner = im;

			im->pkeypoints.push_back(pk);


			fread(&x,sizeof(float),1,f); //x
		}

		imgs_features.push_back(im);
		fclose(f);
	}
	
}

void save_imgMatches(vector<imgMatches*> &imgs_features, string path, vector<string> imgs_name)
{

	for(int i=0;i<imgs_features.size();i++)
	{
		FILE *f = fopen( (path+"/"+imgs_name[i]+".mat").c_str(), "wb");

		fwrite(&imgs_features[i]->rows,sizeof(int),1,f); //
		fwrite(&imgs_features[i]->cols,sizeof(int),1,f); //

		for(int j=0;j<imgs_features[i]->pkeypoints.size();j++)
		{

			fwrite(&imgs_features[i]->pkeypoints[j]->pt.x,sizeof(float),1,f); //x
			fwrite(&imgs_features[i]->pkeypoints[j]->pt.y,sizeof(float),1,f); //y

			fwrite(&imgs_features[i]->pkeypoints[j]->R,sizeof(unsigned char),1,f); //R
			fwrite(&imgs_features[i]->pkeypoints[j]->G,sizeof(unsigned char),1,f); //G
			fwrite(&imgs_features[i]->pkeypoints[j]->B,sizeof(unsigned char),1,f); //B
		}

		fclose(f);
	}
	

}

void load_epgraph(Mat &G, string path, vector<string> imgs_name)
{
	FILE *f = fopen( (path+"/epipolar.graph").c_str(), "rb");

	unsigned int i, j;
	double g;

	fread(&i,sizeof(unsigned int),1,f);
	while(!feof(f))
	{
		
		fread(&j,sizeof(unsigned int),1,f);
		fread(&g,sizeof(double),1,f);

		G.at<double>(i,j) = g;

		fread(&i,sizeof(unsigned int),1,f);
	}

	fclose(f);
			

}

void save_epgraph(Mat G, string path, vector<string> imgs_name)
{

	FILE *f = fopen( (path+"/epipolar.graph").c_str(), "wb");

	//cout<<"SALVANDO G:"<<endl<<G<<endl;

	for(unsigned int i=0;i< G.rows;i++)
		for(unsigned int j=0;j< G.cols;j++)
		{
			if(G.at<double>(i,j) >0)
			{
				fwrite(&i,sizeof(unsigned int),1,f);
				fwrite(&j,sizeof(unsigned int),1,f);
				fwrite(&(G.at<double>(i,j)),sizeof(double),1,f);
			}

		}

		fclose(f);
}

void load_pairMatch(vector<vector<pairMatch*> > &pairs, string path, vector<string> imgs_name, Mat G)
{

	float x,y;
	long idx_cam;

	for(unsigned int i=0;i<G.rows;i++)
		for(unsigned int j=i+1;j<G.cols;j++)
		{
				if(G.at<double>(i,j) > 0)
				{
					FILE *f = fopen( (path+"/" + imgs_name[i]+"_"+imgs_name[j]+".match").c_str(), "rb");

					pairs[i][j] = new pairMatch;

					fread(&pairs[i][j]->homog_inliers, sizeof(pairs[i][j]->homog_inliers),1,f); //homography inliers

					
					fread(&idx_cam, sizeof(idx_cam),1,f); //idx_1
					while(!feof(f))
					{
						

						
						fread(&x, sizeof(x),1,f); //point_1_x
						fread(&y, sizeof(y),1,f); //point_1_y

						pairs[i][j]->idx_cam1.push_back(idx_cam);
						pairs[i][j]->cam1.push_back(Point2f(x,y));

						fread(&idx_cam, sizeof(idx_cam),1,f); //idx_2
						fread(&x, sizeof(x),1,f); //point_2_x
						fread(&y, sizeof(y),1,f); //point_2_y

						pairs[i][j]->idx_cam2.push_back(idx_cam);
						pairs[i][j]->cam2.push_back(Point2f(x,y));
					

						fread(&idx_cam, sizeof(idx_cam),1,f); //idx_1
					}

					fclose(f);
				}


		}

}

void save_pairMatch(vector<vector<pairMatch*> > pairs, string path, vector<string> imgs_name, Mat G)
{
	for(unsigned int i=0;i<G.rows;i++)
		for(unsigned int j=i+1;j<G.cols;j++)
		{
			if(G.at<double>(i,j) > 0)
			{
				FILE *f = fopen( (path+"/" + imgs_name[i]+"_"+imgs_name[j]+".match").c_str(), "wb");


				fwrite(&pairs[i][j]->homog_inliers, sizeof(pairs[i][j]->homog_inliers),1,f); //homography inliers

				for(int k=0;k<pairs[i][j]->cam1.size();k++)
				{
					

					fwrite(&pairs[i][j]->idx_cam1[k], sizeof(pairs[i][j]->idx_cam1[k]),1,f); //idx_1
					fwrite(&pairs[i][j]->cam1[k].x, sizeof(pairs[i][j]->cam1[k].x),1,f); //point_1_x
					fwrite(&pairs[i][j]->cam1[k].y, sizeof(pairs[i][j]->cam1[k].y),1,f); //point_1_y

					fwrite(&pairs[i][j]->idx_cam2[k], sizeof(pairs[i][j]->idx_cam2[k]),1,f); //idx_2
					fwrite(&pairs[i][j]->cam2[k].x, sizeof(pairs[i][j]->cam2[k].x),1,f); //point_2_x
					fwrite(&pairs[i][j]->cam2[k].y, sizeof(pairs[i][j]->cam2[k].y),1,f); //point_2_y



				}

				fclose(f);
			}


		}
}

void load_img_list(string path, vector<string> &imgs_name)
{
	ifstream in (path+"/image.list");

	string l;

	in>>l;
	while(!in.eof())
	{
		imgs_name.push_back(string(l));
		in>>l;
	}

	in.close();
}

void save_img_list(string path, vector<string> imgs_name)
{
	ofstream in (path+"/image.list");

	for(int i=0;i<imgs_name.size();i++)
		in<<imgs_name[i]<<endl;


	in.close();
}

