###################################################################
#                                                                 #
#    High-for-Low (HFL) Boundary Detector                         #
#    Gedas Bertasius gberta-(at)-seas-(dot)-upenn-(dot)-edu       #
#                                                                 #
###################################################################

1. Introduction.

Here we present a High-for-Low (HFL) Boundary Detector documentation. Our method produces semantic boundaries and outperforms state-of-the-art methods in boundary detection on BSDS500 dataset (as of 04/20/2015).

@InProceedings{gberta_2015_ICCV,
author = {Gedas Bertasius and Jianbo Shi and Lorenzo Torresani},
title = {High-for-Low and Low-for-High:
Efficient Boundary Detection from Deep Object Features and its Applications to High-Level Vision},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}


###################################################################

2. Installation Requirements.

a) Caffe Deep Learning library and its Python Wrapper (http://caffe.berkeleyvision.org/installation.html)

All of the required files are already included in the source code. However, Caffe and its python wrapper needs to be compiled according to the instructions in http://caffe.berkeleyvision.org/installation.html. 

c) OpenCV 3.0 (source code is included in this package)
 
Open CV needs to be compiled using the following instructions:

cd PATH/TO/opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/path/to/local/install/directory/ -D WITH_CUDA=OFF ..
make -j8
make install

d) Compiling SE_detector.cpp

1) Go to the directory where SE_detector.cpp is located 
2) Type the following into the command line: g++ SE_detector.cpp -o SE_detector -L/PATH/TO/OPENCV/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_ximgproc -lopencv_core -lopencv_imgproc -I/PATH/TO/OPECNV/include



###################################################################

3. Getting Started.

- Download the VGG model from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md. The model file then needs to be stored in 'examples/VGG/' and should be named as 'VGG_model'

- From the root directory, go to ‘caffe/examples/HFL_detector/‘

- There are four versions of HFL detector: 
  1) ‘HFL_demo_cpu_fast.py’: CPU version, that predicts boundaries fast but with slightly lower performance quality.
  2) ‘HFL_demo_cpu_accurate.py’: CPU version, that predicts boundaries with good performance quality but at a slightly lower speed.
  3) ‘HFL_demo_gpu_fast.py’: GPU version, that predicts boundaries fast but with slightly lower performance quality.
  4) 'HFL_demo_gpu_fast.py’: GPU version, that predicts boundaries with good performance quality but at a slightly lower speed.

-In each of these files you need to specify the Caffe root directory path. (Line 157 and Line 225 in CPU and GPU versions respectively)

-To run HFL detector type "python HFL_demo_cpu_fast.py <image_file_name> <output_file_name>” in the command line.


###################################################################

4. Misc. Notes:

-For highest speed it is recommended to run the HFL detector on GPU. However, it can also be used with CPUs.
 
-The network should be cached in memory for higher efficiency.

-Due to some implementation differences, the results achieved by this HFL version and the results presented in our ICCV paper are a bit different.


###################################################################
