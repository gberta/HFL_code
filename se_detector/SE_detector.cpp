#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>

#include <stdio.h>      /* printf */
#include <math.h> 

#include "wrappers.hpp"
#include <string.h>
#include "sse.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

// convolve one column of I by a [1 p 1] filter (uses SSE)
void convTri1Y( float *I, float *O, int h, float p, int s ) {
  #define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
  int j=0, k=((~((size_t) O) + 1) & 15)/4, h2=(h-1)/2;
  if( s==2 ) {
    for( ; j<k; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
    for( ; j<h2-4; j+=4 ) STR(O[j],_mm_shuffle_ps(C4(2,1),C4(2,5),136));
    for( ; j<h2; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
    if( h%2==0 ) O[j]=I[2*j]+(1+p)*I[2*j+1];
  } else {
    O[j]=(1+p)*I[j]+I[j+1]; j++; if(k==0) k=(h<=4) ? h-1 : 4;
    for( ; j<k; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
    //std::cout << "31" << "\n";
    for( ; j<h-4; j+=4 ) STR(O[j],C4(1,0));
    //std::cout << "33" << "\n";
    for( ; j<h-1; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
    //std::cout << "35" << "\n";
    O[j]=I[j-1]+(1+p)*I[j];
    //std::cout << "37" << "\n";

  }
  #undef C4
}



// convolve I by a [1 p 1] filter (uses SSE)
void convTri1( float *I, float *O, int h, int w, int d, float p, int s ) {
  const float nrm = 1.0f/((p+2)*(p+2)); int i, j, h0=h-(h%4);
  float *Il, *Im, *Ir, *T=(float*) alMalloc(h*sizeof(float),16);
  for( int d0=0; d0<d; d0++ ) for( i=s/2; i<w; i+=s ) {
    Il=Im=Ir=I+i*h+d0*h*w; if(i>0) Il-=h; if(i<w-1) Ir+=h;
    //std::cout << "51" << "\n";
    for( j=0; j<h0; j+=4 ){
      //std::cout << "IM[j] = " << Im[j] <<  "\n";
      //std::cout << "Ir[j] = " << Ir[j] <<  "\n";
      //std::cout << "Il[j] = " << Il[j] <<  "\n";
      STR(T[j],MUL(nrm,ADD(ADD(LDu(Il[j]),MUL(p,LDu(Im[j]))),LDu(Ir[j]))));
    }
      //std::cout << "55" << "\n";
    for( j=h0; j<h; j++ ) T[j]=nrm*(Il[j]+p*Im[j]+Ir[j]);

    //std::cout << "d0 = " << d0 << " i = " << i << " j = " << j << "\n";
    convTri1Y(T,O,h,p,s); O+=h/s;
    //std::cout << "57" << "\n";

  }
  alFree(T);
}




inline float interp( Mat I, int h, int w, float x, float y ) {
	x = x<0 ? 0 : (x>w-1.001 ? w-1.001 : x);
	y = y<0 ? 0 : (y>h-1.001 ? h-1.001 : y);
	int x0=int(x), y0=int(y), x1=x0+1, y1=y0+1;
	float dx0=x-x0, dy0=y-y0, dx1=1-dx0, dy1=1-dy0;

	/*std::cout << "x = " << x << " y = " << y << "\n";
	std::cout << "x0 = " << x0 << " y0 = " << y0 << "\n";
	std::cout << "x1 = " << x1 << " y0 = " << y0 << "\n";
	std::cout << "x0 = " << x0 << " y1 = " << y1 << "\n";
	std::cout << "x1 = " << x1 << " y1 = " << y1 << "\n";
	std::cout << " ##### " << "\n";*/


	//return I[x0*h+y0]*dx1*dy1 + I[x1*h+y0]*dx0*dy1 + I[x0*h+y1]*dx1*dy0 + I[x1*h+y1]*dx0*dy0;
	return I.at<float>(y0,x0)*dx1*dy1 + I.at<float>(y0,x1)*dx0*dy1 + I.at<float>(y1,x0)*dx1*dy0 + I.at<float>(y1,x1)*dx0*dy0;

}

int main( int argc, const char** argv )
    {
        
        std::string modelFilename = argv[1];
        std::string inFilename = argv[2];
        std::string outFilename = argv[3];
    
	//Mat image = imread(inFilename.c_str(), 1);
        cv::Mat image = cv::imread(inFilename.c_str(), 1);
        if ( image.empty() )
        {
            printf("Cannot read image file: %s\n", inFilename.c_str());
            return -1;
        }
 
	int rows = image.rows;
	int cols = image.cols;

	cv::Mat image_sm, image_lrg;


	cv::resize(image, image_sm, Size(cols/2.0,rows/2.0));
	cv::resize(image, image_lrg, Size(cols*2.0,rows*2.0));

	/*cvtColor( image, image, CV_BGR2GRAY );
	cvtColor( image_sm, image_sm, CV_BGR2GRAY );
	cvtColor( image_lrg, image_lrg, CV_BGR2GRAY );*/



	//Gradient
	
	Mat blur_image;
	GaussianBlur( image,  blur_image, Size(3,3), 0, 0, BORDER_DEFAULT );

	

	Mat src_gray;
	cvtColor( blur_image, src_gray, CV_BGR2GRAY );

	 int scale = 1;
	 int delta = 0;
	 int ddepth = CV_32F;
	   
	 //    /// Generate grad_x and grad_y
	 Mat grad_x, grad_y;
	       
	 // Gradient X
	 Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	 //Sobel( src_gray, grad_x, ddepth, 1, 0, 10, scale, delta, BORDER_DEFAULT );
	 //convertScaleAbs( grad_x, abs_grad_x );
	 
	 //Gradient Y
	 Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	 //Sobel( src_gray, grad_y, ddepth, 0, 1, 10, scale, delta, BORDER_DEFAULT );
	 //convertScaleAbs( grad_y, abs_grad_y );


        image.convertTo(image, cv::DataType<float>::type, 1/255.0);
    	image_sm.convertTo(image_sm, cv::DataType<float>::type, 1/255.0);
	image_lrg.convertTo(image_lrg, cv::DataType<float>::type, 1/255.0);

	
/*

	//Sharpening
	
	int d=3;
	float p=2;
	int r=2;
	int s=1;

	//cv::Mat conv_image(image.size(), image.type());
	//float* float_image = (float*)image.data;
	float* float_image=(float*)malloc(rows*cols*d*sizeof(float));
	int counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<cols;i++){
		for (int j=0;j<rows;j++){
		    float_image[counter]=image.at<Vec3f>(j,i)[k];
		    counter++;
		}
	   }
	}

	float* conv_image=(float*)malloc(rows*cols*d*sizeof(float));
	convTri1(float_image,conv_image,rows,cols,d,p,s);
	//exit(1);


	//std::cout << "small rows = " << image_sm.size().height<< "\n";
	//std::cout << "small cols = " << image_sm.size().width<< "\n";

	//cv::Mat conv_image_sm(image_sm.size(), image_sm.type());
	float* float_image_sm=(float*)malloc(image_sm.size().height*image_sm.size().width*d*sizeof(float));
	counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<image_sm.size().width;i++){
		for (int j=0;j<image_sm.size().height;j++){
		    float_image_sm[counter]=image_sm.at<Vec3f>(j,i)[k];
		    counter++;
		}
	   }
	}

	float* conv_image_sm=(float*)malloc(image_sm.size().height*image_sm.size().width*d*sizeof(float));
	convTri1(float_image_sm,conv_image_sm,image_sm.size().height,image_sm.size().width,d,p,s);
	//exit(1);


	//cv::Mat conv_image_lrg(image_lrg.size(), image_lrg.type());
	float* float_image_lrg=(float*)malloc(rows*2*cols*2*d*sizeof(float));
	counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<cols*2;i++){
		for (int j=0;j<rows*2;j++){
		    float_image_lrg[counter]=image_lrg.at<Vec3f>(j,i)[k];
		    counter++;
		}
	   }
	}
	float* conv_image_lrg=(float*)malloc(rows*2*cols*2*d*sizeof(float));
	convTri1(float_image_lrg,conv_image_lrg,rows*2,cols*2,d,p,s);
	//exit(1);
	

	cv::Mat input_image(image.size(), image.type());
	cv::Mat input_image_sm(image_sm.size(), image_sm.type());
	cv::Mat input_image_lrg(image_lrg.size(), image_lrg.type());

	//cv::Mat input_image(image.size(), CV_8UC3);

	//const int mySize[3]={3,rows,cols};
	//cv::Mat input_image = Mat::zeros(rows,cols,CV_32F);
	

	counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<image_sm.size().width;i++){
		for (int j=0;j<image_sm.size().height;j++){
		    input_image_sm.at<Vec3f>(j,i)[k]=conv_image_sm[counter];
		    counter++;
		}
	   }
	}

	counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<cols;i++){
		for (int j=0;j<rows;j++){
		    input_image.at<Vec3f>(j,i)[k]=conv_image[counter];
		    counter++;
		}
	   }
	}

	
	counter=0;
	for (int k=0;k<d;k++){
	   for (int i=0;i<image_lrg.size().width;i++){
		for (int j=0;j<image_lrg.size().height;j++){
		    input_image_lrg.at<Vec3f>(j,i)[k]=conv_image_lrg[counter];
		    counter++;
		}
	   }
	}

	*/


	//cv::imwrite(outFilename, 255*input_image);
	//cv::imwrite(outFilename, 255*image);

	//exit(1);

	////////////

        cv::Mat edges(image.size(), image.type());
    
        cv::Ptr<StructuredEdgeDetection> pDollar =
            createStructuredEdgeDetection(modelFilename);
        //pDollar->detectEdges(input_image, edges);
	pDollar->detectEdges(image, edges);


	//small
        cv::Mat edges_sm(image_sm.size(), image_sm.type());
    
        //pDollar->detectEdges(input_image_sm, edges_sm);
	pDollar->detectEdges(image_sm, edges_sm);

  
	//large
        cv::Mat edges_lrg(image_lrg.size(), image_lrg.type());   
        //pDollar->detectEdges(input_image_lrg, edges_lrg);
	pDollar->detectEdges(image_lrg, edges_lrg);


	cv::resize(edges_sm, edges_sm, image.size());
	cv::resize(edges_lrg, edges_lrg, image.size());




	Mat O=cv::Mat::zeros(rows, cols, CV_32F);
	for(int j=0;j<rows;j++) 
	{
	  for (int i=0;i<cols;i++)
	  {

		O.at<float>(j,i) = atan2(grad_y.at<float>(j,i),grad_x.at<float>(j,i));

		//multiscale edge predictions
		edges.at<float>(j,i)=(edges.at<float>(j,i)+edges_sm.at<float>(j,i)+edges_lrg.at<float>(j,i))/3.0;
		
		//edges.at<float>(j,i)=edges.at<float>(j,i);


	  }
	}


	//NORMALIZATION
	//
	float p=2;
	int s=1;
	

	float* float_edges=(float*)malloc(rows*cols*sizeof(float));
	int counter=0;
	   for (int i=0;i<cols;i++){
		for (int j=0;j<rows;j++){
		    float_edges[counter]=edges.at<float>(j,i);
		    counter++;
		}
	   }
	
	float* conv_edges=(float*)malloc(rows*cols*sizeof(float));
	convTri1(float_edges,conv_edges,rows,cols,1,p,s);

	counter=0;
	
	   for (int i=0;i<image.size().width;i++){
		for (int j=0;j<image.size().height;j++){
		    edges.at<float>(j,i)=conv_edges[counter];
		    counter++;
		}
	   }


       	//cv::imwrite(outFilename, 255*edges);
	//exit(1);





	//NMS
	  
	Mat E=cv::Mat::zeros(rows, cols, CV_32F);
	float m=1.01;
	int r=1;
	s=5;


	std::cout << "Rows = " << rows << " Cols = " << cols << "\n";

	int count=1;
	for( int x=0; x<cols; x++ ){ 
	      //std::cout << "Col = " << x << "\n";
	      for( int y=0; y<rows; y++ ) {
		 //float e=E[x*h+y]=E0[x*h+y]; 
		 float e=edges.at<float>(y,x); 
		 //E.at<uchar>(y,x)=edges.at<uchar>(y,x); 
		 E.at<float>(y,x)=e; 

		 count++;

		 if(e>0){
		 
		  e*=m;
		  float cur_o=O.at<float>(y,x);
		  float coso=cos(cur_o), sino=sin(cur_o);
		
		  //std::cout << "cur_o = " << cur_o << " coso = " << coso << " sino =" << sino <<"\n";

		  for( int d=-r; d<=r; d++ ) if( d ) {
			float e0 = interp(edges,rows,cols,x+d*coso,y+d*sino);
			//float e0=0;

			if(e < e0) { E.at<float>(y,x)=0; break; }
		  }
		 }
	      }
	}


	//std::cout << "Count = " << count << "\n";

	    // suppress noisy edge estimates near boundaries
	   
	   /*s=s>w/2?w/2:s; s=s>h/2? h/2:s;
	   for( int x=0; x<s; x++ ) for( int y=0; y<h; y++ ) {
	      E[x*h+y]*=x/float(s); E[(w-1-x)*h+y]*=x/float(s); }
	      for( int x=0; x<w; x++ ) for( int y=0; y<s; y++ ) {
	    	 E[x*h+y]*=y/float(s); E[x*h+(h-1-y)]*=y/float(s); }
	   */



        if ( outFilename == "" )
        {
            cv::namedWindow("edges", 1);
            cv::imshow("edges", edges);
    
            cv::waitKey(0);
        }
        else
            //cv::imwrite(outFilename, 255*edges);
	    //cv::imwrite(outFilename, dst);
    	    cv::imwrite(outFilename, 255*E);

        return 0;
}
