
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <stdint.h>

using namespace cv;
using namespace std;




int main( int argc, char** argv )
{

	RotatedRect bbox;
	int area 	  = 0;
	int perimeter 	  = 0;
	int num_holes 	  = 0;
	int holes_area 	  = 0;
	float stroke_mean = 0;
	float stroke_std  = 0;

	Mat bw = imread(argv[1], 0);
	threshold( bw, bw, 128, 255, THRESH_BINARY );

	Mat tmp;
	distanceTransform(bw, tmp, CV_DIST_L1,3); //L1 gives distance in round integers while L2 floats

	Scalar mean,std;
        meanStdDev(tmp,mean,std,bw);
	stroke_mean = mean[0];
	stroke_std  = std[0];

	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	findContours( bw, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	area = contourArea(Mat(contours0.at(0)));

    	for (int k=0; k<hierarchy.size();k++)
    	{
	    //TODO check this thresholds, are they coherent? is there a faster way?
	    if ((hierarchy[k][3]==0)&&((((float)contourArea(Mat(contours0.at(k)))/area)>0.01)||(contourArea(Mat(contours0.at(k)))>31)))
	    {
		num_holes++;
		holes_area += (int)contourArea(Mat(contours0.at(k)));
	    }
    	}

	perimeter = (int)contours0.at(0).size();

	bbox = minAreaRect(contours0.at(0));

	fprintf(stdout,"%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", argv[2], stroke_mean, stroke_std, stroke_std/stroke_mean, (float)area, (float)perimeter, (float)perimeter/area, (float)min(bbox.size.width, bbox.size.height)/max(bbox.size.width, bbox.size.height), sqrt(area)/perimeter, (float)num_holes, (float)holes_area/area);

	return(0);
}

