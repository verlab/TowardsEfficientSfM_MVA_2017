
// Copyright (c) 2016 Guilherme POTJE.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef VERLAB_SFM_UTILITARIOS
#define VERLAB_SFM_UTILITARIOS

#define MAX_RES 5000.0 //3280.0 //1280.0 1920.0 2280.0
#define FISHEYE false

#include <string>
#include <sstream>
#include <algorithm>
#include <math.h>

#include <iostream>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/opencv.hpp>

#ifndef WIN32
#include <dirent.h>
#endif

#include <vector>
#include <iostream>
#include <list>
#include <set>


#include <algorithm>


///////

#include <exiv2/exiv2.hpp>
#include <iomanip>
#include <cassert>

#include <stdio.h>
//#include "exif.h"

//#include "Utilitarios.h"


//include the header file
#include "pba/pba.h"

//util.h has several loading/saving functions
#include "pba/util.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define pii pair<int, int>
#define pip pair<int, pii>



struct cloudPoint;
struct imgMatches;
/*struct graphNode
{
    int img_id;
    std::vector<graphNode*> childs;
    std::vector<float> cost;
};
*/


struct pKeypoint
{
      Point2f pt;
      cloudPoint * cloud_p;
      bool checked;
      bool active;
     unsigned char R;
     unsigned char G;
     unsigned char B;
      int master_idx;
      imgMatches *img_owner;
      vector<pair<int,int> > matches; // img_idx, position_idx

};

struct cloudPoint
{

    double x,y,z;
    vector<pKeypoint*> owners;
    bool is_in_cloud;
    long idx;
    double reproj_error;
    char can_ba;
    vector<double> reprj_errs; //to validate the point
    bool added;


};

struct imgMatches
{
  vector<KeyPoint> keypoints_img;
  Mat descriptors_img;
  Matx34d P_mat;
  vector<pKeypoint*> pkeypoints;
  bool triangulated;
  bool already_rmv;
  int rows,cols;
  int id;
  unsigned int qtd_projs;
  int pos;
  bool had_intrinsics;
  bool EXIF;
  bool estimated;


};

struct camera_info
{
  string make;
  string model;
  float sensor_w;
};


const int MAX = 10000;

vector<camera_info> camera_database;

//class implementing Union Find Data Structure with Path Compression
class Union_Find
{
    public:

    int id[MAX], sz[MAX];
    Union_Find(int n)	//class constructor
    {
    for (int i = 0; i < n; ++i)
    {
    id[i] = i;
    sz[i] = 1;
    }
    }
    int root(int i)
    {
    while(i != id[i])
    {
    id[i] = id[id[i]];	//path Compression
    i = id[i];
    }
    return i;
    }
    int find(int p, int q)
    {
    return root(p)==root(q);
    }
    void unite(int p, int q)
    {
    int i = root(p);
    int j = root(q);

    if(sz[i] < sz[j])
    {
    id[i] = j;
    sz[j] += sz[i];
    }
    else
    {
    id[j] = i;
    sz[i] += sz[j];
    }
    }
};


vector< pip > graph;
int n, e=0;
long long int T;

vector<pair<int,int> > Kruskal_MST()
{
    vector<pair <int,int> > arestas;
    Union_Find UF(n);
    int u, v;

    for (int i = e-1; i >= 0; --i)
    {
        u = graph[i].second.first;
        v = graph[i].second.second;
        if( !UF.find(u, v) )
        {
        // printf("uniting %d and %d\n",u,v );
        UF.unite(u, v);
        T += graph[i].first;

        }
        else
        {
            arestas.push_back(graph[i].second);
        }
    }

    return arestas;
}



vector< pair<int,int> > get_Kruskal(Mat G)
{

    for (int i = 0; i < G.rows; i++)
    {
      for(int j = i+1; j<G.cols; j++)
      {

        graph.push_back(pip( G.at<double>(i,j), pii(i,j)));
        e++;
      }
    }

    n = G.rows;
    std::sort(graph.begin(), graph.end());	//sort the edges in increasing order of cost

    T = 0;
    return Kruskal_MST();
    printf("%lld\n",T);
    //return 0;
}

bool sort_response(KeyPoint k1, KeyPoint k2)
{
    return k1.size>k2.size; //response
}

void sort_keypoints_by_response(vector<KeyPoint> &keypoints)
{
    std::sort(keypoints.begin(),keypoints.end(),sort_response);

}

void compute_keypoints(imgMatches *im, Mat imgCena, int qualDescritor)
{
    //imshow("",imgCena); waitKey(0);

    vector<KeyPoint> keypointsCena;
	Mat descriptorsCena;

	if (qualDescritor ==0)
	{   //keypoints SIFT

	    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(pow(imgCena.rows,1.3)*3.85, 3, 0.04, 5, 1.6);
	   // SIFT* detector = new SIFT::SIFT(pow(imgCena.rows,1.3)*3.85, 3, 0.04, 5, 1.3); //0.0065   1.6*0.2  1.2*4.55  1.2*1.55  -- 4.55 sigma 1.7 //6.55 1.85
	    detector->detect(imgCena, keypointsCena);
	    cout<<"SIFT Detected: " << keypointsCena.size() <<" keypoints..."<<endl; ////




	   /* for(double i=0;i<6;i++) //
        {

	    SiftFeatureDetector* detector = new SiftFeatureDetector(pow(imgCena.rows,1.3)*1.85, 3, 0.04, 5, 1.6-(i/7)); //0.0065   1.6*0.2  1.2*4.55  1.2*1.55  -- 4.55 sigma 1.7 //6.55
	    detector->detect(imgCena, keypointsCena);
	    cout<<"<> Found " <<keypointsCena.size()<<" keypoints." <<endl;

	    if(keypointsCena.size()>2000) //If we found more than 2000 keypoints with this sigma
        break;
        }
        */

  sort_keypoints_by_response(keypointsCena);


	}
  else if(qualDescritor == 1) //AKAZE
  {

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detect(imgCena, keypointsCena);
    //akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

  }
	/*else if(qualDescritor == 1)
	{   //keypoints SURF
   	    int minHessian = 400;
	    SurfFeatureDetector *detector = new SurfFeatureDetector(minHessian);
		detector->detect(imgCena, keypointsCena);
	}
	else if(qualDescritor ==2)
	{

	    OrbFeatureDetector* detector = new OrbFeatureDetector(pow(imgCena.rows,1.2)*0.8, 1.2f, 8, 31, 0,2, ORB::HARRIS_SCORE, 31);
	    detector->detect(imgCena, keypointsCena);
	}
	else
	{

    cv::Ptr<FeatureDetector> detector =  FeatureDetector::create("PyramidFAST");
    detector->detect(imgCena, keypointsCena);

	}
	*/



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
	/*else if(qualDescritor ==1)
	{	// SURF
	    SurfDescriptorExtractor *extractor = new SurfDescriptorExtractor();
		extractor->compute(imgCena, keypointsCena, descriptorsCena);

	}
	else if(qualDescritor==2)
	{
        //BriefDescriptorExtractor *extractor = new BriefDescriptorExtractor();
		//extractor->compute(imgCena, keypointsCena, descriptorsCena);

		OrbDescriptorExtractor * extractor = new OrbDescriptorExtractor();
		extractor->compute(imgCena, keypointsCena, descriptorsCena);

	}
	else
	{
	   cv::Ptr<DescriptorExtractor> extractor =  DescriptorExtractor::create("ORB");
	   extractor->compute(imgCena, keypointsCena, descriptorsCena);

	}*/

	im->keypoints_img = keypointsCena;
	im->descriptors_img = descriptorsCena;
	im->triangulated = false;
	im->already_rmv= false;
	im->rows = imgCena.rows;
	im->cols = imgCena.cols;


	for(int i=0;i< im->keypoints_img.size();i++)
	{
	    pKeypoint * pk = new pKeypoint;
	    pk->cloud_p = NULL;
	    //pk->cam_idx = -1;

	    pk->pt= im->keypoints_img[i].pt;
        pk->R = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[2];
        pk->G = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[1];
        pk->B = imgCena.at<Vec3b>(pk->pt.y, pk->pt.x)[0];

	pk->img_owner = im;

	    im->pkeypoints.push_back(pk);
	}

   // imgCena.release();
}


cv::Mat correct_contrast(cv::Mat bgr_image)
{
    cv::Mat lab_image;
    cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

   // convert back to RGB
   cv::Mat image_clahe;
   cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

   return image_clahe;
}



cv::Mat open_cv_image(string path, double downscale_factor)
{
  cv::Mat m_;

  //m_ = correct_contrast(imread(path));
  m_ = imread(path);

 if(m_.rows==0 || m_.cols==0) //error reading the image
  return m_;


    double maior, menor;

    if(downscale_factor==1.0)
    {
        if(m_.rows>m_.cols)
        {
          maior = m_.rows; menor=m_.cols;
        }
        else
        {
          maior = m_.cols; menor = m_.rows;
        }

          if(MAX_RES/maior < 1.0) //too big image, resize to x
        {
            downscale_factor = MAX_RES/maior;
        }

    }

    if(downscale_factor != 1.0)
      cv::resize(m_,m_,Size(),downscale_factor,downscale_factor);

  return m_;
}





cv::Mat open_cv_image(string path, double downscale_factor, double &scale_factor)
{
  cv::Mat m_;

  //m_ = correct_contrast(imread(path));
  m_ = imread(path);

 if(m_.rows==0 || m_.cols==0) //error reading the image
  return m_;


    double maior, menor;

    if(downscale_factor==1.0)
    {
        if(m_.rows>m_.cols)
        {
          maior = m_.rows; menor=m_.cols;
        }
        else
        {
          maior = m_.cols; menor = m_.rows;
        }

          if(MAX_RES/maior < 1.0) //too big image, resize to x
        {
            downscale_factor = MAX_RES/maior;
        }

    }

    if(downscale_factor != 1.0)
    {
      cv::resize(m_,m_,Size(),downscale_factor,downscale_factor);
      scale_factor = downscale_factor;
    }

  return m_;
}

double read_exif_gps(double &lat, double &longit, string img_src)
{
  try {

      //cout<<"NAME "<<img_src<<endl;
      lat =0; longit = 0;
      double alt = 0;

      Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(img_src.c_str());
      assert(image.get() != 0);
      image->readMetadata();

      Exiv2::ExifData &exifData = image->exifData();
      if (exifData.empty()) {
          std::string error(img_src.c_str());
          error += ": No Exif data found in the file"; cout<<error<<endl;
         // throw Exiv2::Error(1, error);
         lat =0; longit=0; return 0;
      }
      Exiv2::ExifData::const_iterator end = exifData.end();
      for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i)
      {
          const char* tn = i->typeName();
         /* std::cout << std::setw(44) << std::setfill(' ') << std::left
                    << i->key() << " "
                    << "0x" << std::setw(4) << std::setfill('0') << std::right
                    << std::hex << i->tag() << " "
                    << std::setw(9) << std::setfill(' ') << std::left
                    << (tn ? tn : "Unknown") << " "
                    << std::dec << std::setw(3)
                    << std::setfill(' ') << std::right
                    << i->count() << "  "
                    << std::dec << i->value()
                    << "\n";*/
                //  getchar();
                string a = i->key();
                //cout<<a<<endl;


                if(a.find("GPSInfo")!=std::string::npos && a.find("GPSLatitude")!=std::string::npos && a.find("GPSLatitudeR")==std::string::npos)
                {


                  lat = i->value().toFloat(0)+i->value().toFloat(1)/60.0+i->value().toFloat(2)/3600.0;
                // cout<<"ACHOOOOOO "<<" "<< lat <<endl;

                }


                if(a.find("GPSInfo")!=std::string::npos && a.find("GPSLongitude")!=std::string::npos && a.find("GPSLongitudeR")==std::string::npos)
                {

                  longit = i->value().toFloat(0)+i->value().toFloat(1)/60.0+i->value().toFloat(2)/3600.0;
                // cout<<"ACHOOOOOO "<<" "<< longit <<endl;

                }

                 if(a.find("GPSInfo")!=std::string::npos && a.find("GPSAltitude")!=std::string::npos && a.find("GPSAltitudeR")==std::string::npos)
                {

                  alt = i->value().toFloat(0);//+i->value().toFloat(1)/60.0+i->value().toFloat(2)/3600.0;
                // cout<<"ACHOOOOOO "<<" "<< longit <<endl;

                }

      }

      return alt;
  }
  catch (Exiv2::AnyError& e) {
      std::cout << "Caught Exiv2 exception '" << e.what() << "'\n";
  //    return -1;
  }

}


//////////////SPLIT STRING//////////////

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

/////////////////////////////////////////


void load_camera_db()
{
   ifstream in ("/home/vnc/Dropbox/Projetos/Guilherme/sensor_width_camera_database.txt");


  while(!in.eof())
  {

   camera_info ci;
   string aux;

   getline(in,aux);
   if(aux.length()>1)
   {
     vector<string> parts = split(aux,';');

     ci.make = parts[0];
     ci.model = parts[1];

     ci.sensor_w = std::stof(parts[2]);

     std::transform(ci.make.begin(), ci.make.end(), ci.make.begin(), ::tolower);

     std::transform(ci.model.begin(), ci.model.end(), ci.model.begin(), ::tolower);

     camera_database.push_back(ci);
 }

  }

}


float find_sensor_database(string cam_make, string cam_model)
{

  float sw =0;


  for(int i=0;i<camera_database.size();i++)
  {
    if(camera_database[i].make.find(cam_make)!=std::string::npos)
    {

      vector<string> parts = split(cam_model,' ');
      int cont=0;

      for(int j=0;j<parts.size();j++)
        if(camera_database[i].model.find(parts[j])!=std::string::npos)
          cont++;

      if(cont==parts.size())
      {

        sw = camera_database[i].sensor_w;
        break;

      }
    }
  }

  return sw;
}

bool find_string(vector<string> parts, string query)
{

  for(int i=0;i<parts.size();i++)
  {

    if(parts[i] ==query)
    {
      return true;
      break;
    }
  }

  return false;
}

bool mount_K_EXIF(cv::Mat &K_new, string img_src, imgMatches* img)
{
  double focal_35mm=0;
  double focal = 0;
  double focal_pixel = 0;
  double sensor_size=0;
  double focalplane_x=0;


  string camera_make;
  string camera_model;

  try {

   //cout<<"NAME "<<img_src<<endl;


      Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(img_src.c_str());
      assert(image.get() != 0);
      image->readMetadata();

      Exiv2::ExifData &exifData = image->exifData();
      if (exifData.empty()) {
          std::string error(img_src.c_str());
          error += ": No Exif data found in the file"; cout<<error<<endl;
         // throw Exiv2::Error(1, error);
         return false;
      }
      Exiv2::ExifData::const_iterator end = exifData.end();
      for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i)
      {
          const char* tn = i->typeName();

                string a = i->key(); //searching by TAG
                //vector<string> parts = split(a,'.');

                //cout<<"Key value as string: "<<a<<endl; getchar();

                if(a.find("FocalLength")!=std::string::npos && a.find("FocalLengthIn35mmFilm") == std::string::npos)
                //if(find_string(parts,string("FocalLength")))
                {


                 // i->value().toFloat(0)+i->value().toFloat(1)/60.0+i->value().toFloat(2)/3600.0;
                  if(focal==0)
                   focal = i->value().toFloat(0);
                 //cout<<"ACHOOOOOO focal = "<<" "<< focal <<endl;

                // if(img_src.find("jayniebell_2112884556.jpg") != std::string::npos)
                //{cout<<"Focal found = " <<focal<<endl; }
                // cout<<a<<endl;

                }


                if(a.find("FocalLengthIn35mmFilm")!=std::string::npos)
                //if(find_string(parts,string("FocalLengthIn35mmFilm")))
                {

                 focal_35mm = i->value().toFloat(0);
                 //cout<<"ACHOOOOOO focal_35mm = "<<" "<< focal_35mm <<endl;

                }

                if(a.find("FocalPlaneXResolution")!=std::string::npos)
                //if(find_string(parts,string("FocalPlaneXResolution")))
                {

                 focalplane_x = i->value().toFloat(0);
                 //cout<<"ACHOOOOOO focal_35mm = "<<" "<< focal_35mm <<endl;

                }


                if(a.find("Image.Make")!=std::string::npos)
                {
                 camera_make = i->value().toString(0);
                 std::transform(camera_make.begin(), camera_make.end(), camera_make.begin(), ::tolower);
                  //cout<<"ACHOOOOOO make = "<<" "<< camera_make <<endl;
                }

                if(a.find("Image.Model")!=std::string::npos)
                {
                  camera_model =  i->value().toString(0);
                  std::transform(camera_model.begin(), camera_model.end(), camera_model.begin(), ::tolower);
                   //cout<<"ACHOOOOOO model = "<<" "<< camera_model <<endl;
                }

      }


   if(camera_make.length()>0 && camera_model.length()>0)
   sensor_size = find_sensor_database(camera_make,camera_model);


      if( focal != 0 && sensor_size > 0)
      {


          double major = (img->cols>img->rows)? img->cols : img->rows;

          focal_pixel = (focal*major)/(double)sensor_size;

          //cout<<"before= "<<K_new.at<double>(0,0)<<" "<<endl;
          K_new.at<double>(0,0) = focal_pixel;
          K_new.at<double>(1,1) = focal_pixel;
          //cout<<"estimated f = "<<focal_pixel<<endl; getchar();

         /* if(img_src.find("jayniebell_2112884556.jpg") != std::string::npos)
          {
            cout<<"sensor size =" <<sensor_size<<endl;
            cout<<"major = " <<major<<endl;
            cout<<"focal ="<<focal<<endl;
            getchar();

          }
        */
         // cout<<"Found F_length = "<<focal_pixel<<endl;
         // cout<<"focal mm = "<<focal<<endl;

         // getchar();
          return true;

      }
      else if(focal_35mm!=0) // find focal in pixels
      {

        focal_pixel = (focal_35mm*sqrt(img->cols*img->cols + img->rows*img->rows))/43.27;

        K_new.at<double>(0,0) = focal_pixel;
        K_new.at<double>(1,1) = focal_pixel;


        return true;
      }
       else if(focalplane_x!=0 && focal != 0)
      {
        double major = (img->cols>img->rows)? img->cols : img->rows;

        sensor_size = (major/focalplane_x)*25.4;

        focal_pixel = (focal*major)/(double)sensor_size;

          //cout<<"before= "<<K_new.at<double>(0,0)<<" "<<endl;
          K_new.at<double>(0,0) = focal_pixel;
          K_new.at<double>(1,1) = focal_pixel;

          return true;

      }




    return false;
  }
  catch (Exiv2::AnyError& e) {
      std::cout << "Caught Exiv2 exception '" << e.what() << "'\n";
      return false;
  }

}



/*
int read_gps_data(double &lat, double &longit, string arq)
{
    // Read the JPEG file into a buffer
  FILE *fp = fopen(arq.c_str(), "rb");
  if (!fp) {
    printf("Can't open file.\n");
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  unsigned long fsize = ftell(fp);
  rewind(fp);
  unsigned char *buf = new unsigned char[fsize];
  if (fread(buf, 1, fsize, fp) != fsize) {
    printf("Can't read file.\n");
    delete[] buf;
    return -2;
  }
  fclose(fp);

  // Parse EXIF
  EXIFInfo result;
  int code = result.parseFrom(buf, fsize);
  delete[] buf;
  if (code)
  {
    printf("Error parsing EXIF: code %d\n", code);
    return -3;
  }

   lat = result.GeoLocation.Latitude;
   longit = result.GeoLocation.Longitude;


}
*/

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

bool hasEndingLower (string const &fullString_, string const &_ending)
{
	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
	return hasEnding(fullstring,ending);
}

bool sort_str(string a1, string a2)
{
    return a1<a2;
}

void sort_by_name(vector<string> &files)
{
    std::sort(files.begin(),files.end(),sort_str);
}


void open_imgs_dir(char* dir_name, std::vector<imgMatches*>& images,std::vector<cv::Mat>& imgs, std::vector<std::string>& images_names, double downscale_factor, cv::Mat distortion, bool use_undistort, cv::Mat &K, int qualDescritor) {
	if (dir_name == NULL) {
		return;
	}

	string dir_name_ = string(dir_name);
	vector<string> files_;
  Mat new_K;

#ifndef WIN32
//open a directory the POSIX way

	DIR *dp;
	struct dirent *ep;
	dp = opendir (dir_name);

	if (dp != NULL)
	{
		while (ep = readdir (dp)) {
			if (ep->d_name[0] != '.')
				files_.push_back(ep->d_name);
		}

		(void) closedir (dp);
	}
	else {
		cerr << ("Couldn't open the directory");
		return;
	}

#else
//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;

	if(dir_name_[dir_name_.size()-1] == '\\' || dir_name_[dir_name_.size()-1] == '/') {
		dir_name_ = dir_name_.substr(0,dir_name_.size()-1);
	}

	hFind = FindFirstFile(string(dir_name_).append("\\*").c_str(), &fdata);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(fdata.cFileName, ".") != 0 &&
				strcmp(fdata.cFileName, "..") != 0)
			{
				if (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					continue; // a diretory
				}
				else
				{
					files_.push_back(fdata.cFileName);
				}
			}
		}
		while (FindNextFile(hFind, &fdata) != 0);
	} else {
		cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		cerr << "some other error with opening directory: " << GetLastError() << endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;
#endif
    long cont=1;
    int cont_img =0;
    double scale_factor;

    sort_by_name(files_);
	for (unsigned int i=0; i<files_.size(); i++)
	 {
		if (files_[i][0] == '.' || !(hasEndingLower(files_[i],"jpg")||hasEndingLower(files_[i],"png")  || hasEndingLower(files_[i],"jpeg"))) {
			continue;//
		}
		cv::Mat m_ = open_cv_image(string(dir_name_).append("/").append(files_[i]),downscale_factor);
    if(m_.rows==0 || m_.cols==0) //error reading the image
    {
      continue;

    }
    double maior, menor;
    if(m_.rows>m_.cols)
    {
      maior = m_.rows; menor=m_.cols;
    }
    else
    {
      maior = m_.cols; menor = m_.rows;
    }
    if(maior/menor > 2.0)//aspect ratio validation, if its weird just skip image
      continue;



			//if needs undistort
			if(use_undistort)
			{   
        cv::Mat undistorted;


        if(FISHEYE)
        {
          Mat newCamMat, map1, map2;
          Size imageSize = m_.size();

          fisheye::estimateNewCameraMatrixForUndistortRectify(K, distortion, imageSize,
                                                              Matx33d::eye(), newCamMat, 1.0,imageSize,0.8); //1
          fisheye::initUndistortRectifyMap(K, distortion, Matx33d::eye(), newCamMat, imageSize,
                                           CV_16SC2, map1, map2);

          remap(m_, undistorted, map1, map2, INTER_LINEAR);

          new_K = newCamMat;
        }
        else
        {
               // cout<<"undistorting with "<<distortion<<m_.rows<<" "<<m_.cols<<endl;
			    undistort(m_,undistorted,K,distortion);
			    
        }

      m_ = undistorted;
      char buf[256];
      sprintf(buf, "%s/result/undistorted/%s",dir_name,files_[i].c_str());

			imwrite(buf,m_);

			}
      else
      {
        char buf[256];
        sprintf(buf, "%s/result/undistorted/%s",dir_name,files_[i].c_str());
        imwrite(buf,m_);
      }

		images_names.push_back(files_[i]);

        imgs.push_back(m_);





	}

  K = new_K;

}



bool read_params(double &img_factor, double &meters_threshold, double &qtd_inliers, double &percentile, bool &show_final_result, bool &flann, int &skip_ba, int &minimum_matches, char *img_path, char *input_file, double &ss_w, double &ss_h, double &focal, vector<double> &flen)
{

 if(strlen(input_file)<3) //
 {
    strcpy(input_file,"/local/Datasets/praca_icex/sfm_params.txt");
 }

 ifstream in (input_file);
 string aux;
 int q;



in>>img_factor; getline(in,aux);
in>>meters_threshold; getline(in,aux);
in>>qtd_inliers; getline(in,aux);
in>>percentile; getline(in,aux);
in>>q; getline(in,aux);
if(q==0)
show_final_result=false;
else
show_final_result=true;
in>>q; getline(in,aux);
if(q==0)
flann=false;
else
flann=true;
in>>skip_ba; getline(in,aux);
in>>minimum_matches; getline(in,aux);
in>>img_path; getline(in,aux);
in>>ss_w>>ss_h; getline(in,aux);
in>>focal; getline(in,aux);

aux.clear();
in>>aux;

cout<<img_path<<" Correct ? "<<endl<<endl; //getchar();

if(aux.length()>0) //read the txt containing the focal length params
{
    ifstream in2 (aux.c_str());

    string imgname;
    double focal;
    int idx;

    int i=0;

    while(!in2.eof())
    {
       // in2>>idx;
        in2>>imgname;
        in2>>focal;

        flen.push_back(focal);
    }
}

in.close();

}

#endif
