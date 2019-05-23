cmake_minimum_required(VERSION 2.8)
project(VerLab_SFM)
find_package(OpenCV 2.4 REQUIRED)
find_package(Ceres REQUIRED)

#find_package(PCL 1.7 REQUIRED)
find_package(PkgConfig)

#pkg_check_modules(EIGEN3 REQUIRED eigen3)
#include_directories( ${EIGEN3_INCLUDE_DIRS} )

#include_directories( ${PCL_INCLUDE_DIRS} )
#link_directories( ${PCL_LIBRARY_DIRS} )
#add_definitions ( ${PCL_DEFINITIONS} )

include_directories( ~/Sources/vsfm/pba/src)
link_directories( ~/Sources/vsfm/pba/bin)

FIND_PACKAGE(OpenMVG REQUIRED)
INCLUDE_DIRECTORIES(${OPENMVG_INCLUDE_DIRS})
#link_directories( ~/Gabriel/Verlab_SFM/openMVG_install/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -g -pthread")

INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

file(GLOB SFM_LIB
	codigos/BundleAdjustment.*
	codigos/Utilitarios.*
	codigos/VocabTree.*
	codigos/AC_Ransac.*
	codigos/thread_Match.*
	codigos/parallel_Features.*
	codigos/ceres_BA.*
	codigos/ceres_camera_functor.*
	codigos/input_output.*
	codigos/graph_utils.*
)

add_library(VerLab_SFM_LIB ${SFM_LIB})
add_executable (VerLab_SFM main.cpp)
target_link_libraries(VerLab_SFM  
			VerLab_SFM_LIB
			${OpenCV_LIBS}
			${PCL_LIBRARIES}
			exiv2
			cvsba
			libpba_no_gpu.a	
			${CERES_LIBRARIES}
			${OPENMVG_LIBRARIES}		
)

