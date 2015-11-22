# HFL Boundary Detector

This is the High-for-Low (HFL) Boundary Detector documentation. Our method produces semantic boundaries and outperforms state-of-the-art methods in boundary detection on BSDS500 dataset. This work has been published in ICCV 2015 Conference.

Citation:  
@InProceedings{gberta_2015_ICCV,  
author = {Gedas Bertasius and Jianbo Shi and Lorenzo Torresani},  
title = {High-for-Low and Low-for-High: Efficient Boundary Detection from Deep Object Features and its Applications to High-Level Vision},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
month = {December},  
year = {2015}  
}

## Installation

1. Caffe Deep Learning library and its Python Wrapper:

	Caffe source code is included. Caffe and its python wrapper need to be compiled as instructed in http://caffe.berkeleyvision.org/installation.html. 

2. OpenCV-3.0:
 
	OpenCV-3.0 source code is included in this package. OpenCV-3.0 needs to be compiled using the following instructions:

	cd PATH/TO/opencv  
	mkdir release  
	cd release  
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/path/to/local/install/directory/ -D WITH_FFMEPG=OFF -D WITH_CUDA=OFF ..  
	make -j8  
	make install  

3. Compiling SE_detector.cpp:

	g++ SE_detector.cpp -o SE_detector -L/PATH/TO/OPENCV/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_ximgproc -lopencv_core -lopencv_imgproc -I/PATH/TO/OPENCV/include

4. Cudamat:

	To run HFL on GPU, you would need to install Cudamat as well. Download the code from https://github.com/cudamat/cudamat. Then install the software using the command:

	python setup.py install --user

	After the installation is complete, add the Cudamat library path to the PYTHONPATH system variable.


## Usage

First, download the VGG model from http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel. The model file then needs to be stored in 'examples/VGG/' and should be named as 'VGG_model'

Next, from the root directory, go to ‘caffe/examples/HFL_detector/‘. You will find four versions of HFL detector:

1. ‘HFL_demo_cpu_fast.py’: CPU version, that predicts boundaries fast but with slightly lower performance quality.
2. ‘HFL_demo_cpu_accurate.py’: CPU version, that predicts boundaries with good performance quality but at a slightly lower speed.
3. ‘HFL_demo_gpu_fast.py’: GPU version, that predicts boundaries fast but with slightly lower performance quality.
4. 'HFL_demo_gpu_fast.py’: GPU version, that predicts boundaries with good performance quality but at a slightly lower speed.

IMPORTANT: In each of these files you need to specify the Caffe root directory path. (See Lines 144,144,179 and 153 in the 4 versions respectively). Finally, to run HFL detector type:

python HFL_demo_cpu_fast.py input.jpg output.jpg


## Notes

1. For the highest speed it is recommended to run the HFL boundary detector on GPU. However, it can also be used with CPUs.
2. The network should be cached in memory for higher efficiency.
3. Due to some implementation differences, the results achieved by this HFL version and the results presented in our ICCV paper are a bit different.


