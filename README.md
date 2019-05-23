# verlab_sfm

Efficient structure-from-motion pipeline

Dependencies:
  - OpenCV
  - Ceres Solver
  - OpenMVG
  - Exiv2

Please install those libraries before compiling the code

Usage:

 1. Put the sfm_params.txt file in a desired location and change the image dataset path for the desired one.
 2. Create a directory inside the image path called 'result'
 3. Inside the 'result' dir, create the directories 'txt' 'visualize' 'models' 'undistorted'
 4. You can change the sfm_params.txt parameters according one's needs
 5. Call "./VerlabSFM [path_to_sfm_params] 1" for image registration and "./VerlabSFM [path_to_sfm_params] 2" for camera pose and sparse structrure estimation.

Obs: For fisheye distorted images (like ones taken with a GoPro) it is strongly recommended to calibrate the images and remove the distortion before using this pipeline on them.

