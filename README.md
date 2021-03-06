# Project #

This project is based on the paper [Towards an efficient 3D model estimation methodology for aerial and ground images ](https://www.verlab.dcc.ufmg.br/three-dimensional-reconstruction-from-large-image-datasets/) which was published at **Machine Vision and Applications Journal (2017)**. This code implements a Structure-from-Motion reconstruction pipeline described in the paper mentioned.

For more information, please acess the [project page](https://www.verlab.dcc.ufmg.br/three-dimensional-reconstruction-from-large-image-datasets/).



## Contact ##

### Authors ###

* Guilherme Augusto Potje - PhD. Student - UFMG - guipotje@dcc.ufmg.br
* Gabriel Dias Resende - Undergraduate Student (Former member of VeRlab) - UFMG - gabriel.diasresende@gmail.com
* Erickson Rangel do Nascimento - Advisor - UFMG - erickson@dcc.ufmg.br
* Mario Fernando Montenegro Campos - Advisor - UFMG - mario@dcc.ufmg.br

### Institution ###

Federal University of Minas Gerais (UFMG)  
Computer Science Department  
Belo Horizonte - Minas Gerais -Brazil 

### Laboratory ###

![VeRLab](https://www.dcc.ufmg.br/dcc/sites/default/files/public/verlab-logo.png)  

__VeRLab:__ Vison and Robotic Laboratory  
http://www.verlab.dcc.ufmg.br

## Dependencies:
  - OpenCV
  - Ceres Solver
  - OpenMVG
  - Exiv2

### Please install those libraries before compiling the code

### Usage:

 1. Put the sfm_params.txt file in a desired location and change the image dataset path for the desired one.
 2. Create a directory inside the image path called 'result'
 3. Inside the 'result' dir, create the directories 'txt' 'visualize' 'models' 'undistorted'
 4. You can change the sfm_params.txt parameters according one's needs
 5. Call "./VerlabSFM [path_to_sfm_params] 1" for image registration and "./VerlabSFM [path_to_sfm_params] 2" for camera pose and sparse structrure estimation.

Obs: For fisheye distorted images (like ones taken with a GoPro) it is strongly recommended to calibrate the images and remove the distortion before using this pipeline on them.

---

## Citation ##

If you are using it to academic purpose, please cite: 

Potje, G., Resende, G., Campos, M., & Nascimento, E. R. (2017). __Towards an efficient 3D model estimation methodology for aerial and ground images__. Machine Vision and Applications, 28(8), 937-952.

### Bibtex entry ###

> @article{Potje2017,  
> title={Towards an efficient 3D model estimation methodology for aerial and ground images},  
> author={Potje, Guilherme and Resende, Gabriel and Campos, Mario and Nascimento, Erickson R},  
> journal={Machine Vision and Applications},  
> pages={1--16},  
> year={2017},  
> publisher={Springer}  
> doi = {10.1007/s00138-017-0875-x},  
> url = {https://link.springer.com/article/10.1007/s00138-017-0875-x}  
> }  

