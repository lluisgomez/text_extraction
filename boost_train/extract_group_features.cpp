
#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv/highgui.h>

#include <vector>
#include <fstream>
#include <stdint.h>

using namespace cv;
using namespace std;



// Boosted tree classifier for single characters
CvBoost character_classifier;

float classifyRegion( Mat& region, float &_stroke_mean, float &_aspect_ratio, float &_compactness, float &_num_holes, float &_holearea_area_ratio )
{

	RotatedRect bbox;
	int area 	  = 0;
	int perimeter 	  = 0;
	int holes_area 	  = 0;
	float stroke_mean = 0;
	float stroke_std  = 0;
	Mat bw, tmp;
	region.copyTo(bw);
	distanceTransform(bw, tmp, CV_DIST_L1,3); //L1 gives distance in round integers while L2 floats

	Scalar mean,std;
  meanStdDev(tmp,mean,std,bw);
	stroke_mean = mean[0];
	stroke_std  = std[0];

	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	findContours( bw, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	area = contourArea(Mat(contours0.at(0)));

	_num_holes = 0;

    	for (int k=0; k<hierarchy.size();k++)
    	{
	    //TODO check this thresholds, are they coherent? is there a faster way?
	    if ((hierarchy[k][3]==0)&&((((float)contourArea(Mat(contours0.at(k)))/area)>0.01)||(contourArea(Mat(contours0.at(k)))>31)))
	    {
		_num_holes++;
		holes_area += (int)contourArea(Mat(contours0.at(k)));
	    }
    	}

	perimeter = (int)contours0.at(0).size();

	bbox = minAreaRect(contours0.at(0));

	//fprintf(stdout,"X %f %f %f %f %f\n", stroke_std/stroke_mean, (float)min(bbox.size.width, bbox.size.height)/max(bbox.size.width, bbox.size.height), sqrt(area)/perimeter, (float)_num_holes, (float)holes_area/area);

	_stroke_mean 	= stroke_mean;
	_aspect_ratio 	= (float)min(bbox.size.width, bbox.size.height)/max(bbox.size.width, bbox.size.height);
	_compactness  	= sqrt(area)/perimeter;
	_holearea_area_ratio = (float)holes_area/area;


	float arr[] = {0, stroke_mean, stroke_std, stroke_std/stroke_mean, (float)area, (float)perimeter, (float)perimeter/area, _aspect_ratio, _compactness, _num_holes, _holearea_area_ratio};
	vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

	float votes = character_classifier.predict( Mat(sample), Mat(), Range::all(), false, true );
	return votes;
}



int main( int argc, char** argv )
{


	ifstream ifile("./trained_boost_char.xml");
	if (ifile) 
	{
		character_classifier.load("./trained_boost_char.xml", "boost");
	} else {
		fprintf(stderr,"File ./trained_boost_char.xml not found! \n");
		exit(-1);
	}

    	Mat bw = imread(argv[1], 0);

  copyMakeBorder(bw, bw, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(255));
	threshold( bw, bw, 128, 255, THRESH_BINARY_INV ); //group samples are black over white

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	Mat bw2;
	bw.copyTo(bw2);

	findContours( bw2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


	int num_regions = 0;

	for( int i = 0; i < contours.size(); i++ )
    		if ((hierarchy[i][3]==-1))
			num_regions++;

	Mat votes          ( num_regions, 1, CV_32F, 1 );
	Mat stroke_means   ( num_regions, 1, CV_32F, 1 );
	Mat aspect_ratios  ( num_regions, 1, CV_32F, 1 );
	Mat compactnesses  ( num_regions, 1, CV_32F, 1 );
	Mat nums_holes     ( num_regions, 1, CV_32F, 1 );
	Mat holeareas_area ( num_regions, 1, CV_32F, 1 );

	int idx = 0;	
	for( int i = 0; i < contours.size(); i++ )
	{
		if ((hierarchy[i][3]==-1))
		{
		Rect bbox = boundingRect(contours.at(i));

		Mat canvas = Mat::zeros(cvSize(bw.cols, bw.rows),CV_8UC1);
		drawContours( canvas, contours, i, Scalar(255), CV_FILLED, 8, hierarchy );

		Mat region = Mat::zeros(cvSize(bbox.width+20, bbox.height+20),CV_8UC1);

		canvas(bbox).copyTo( region(Rect(10, 10, bbox.width, bbox.height)) );


		float stroke_mean, aspect_ratio, compactness, num_holes, holearea_area_ratio;
		votes.at<float>(idx,0) = classifyRegion(region, stroke_mean, aspect_ratio, compactness, num_holes, holearea_area_ratio);
		stroke_means.at<float>(idx,0) = stroke_mean;
		aspect_ratios.at<float>(idx,0) = aspect_ratio;
		compactnesses.at<float>(idx,0) = compactness;
		nums_holes.at<float>(idx,0) = num_holes;
		holeareas_area.at<float>(idx,0) = holearea_area_ratio;
	idx++;
		}

	}

	Scalar mean,std;
	meanStdDev( votes, mean, std );
	fprintf( stdout, "%s,%f,%f,%f", argv[2], mean[0], std[0], std[0]/mean[0] ); 
	meanStdDev( stroke_means, mean, std );
	fprintf( stdout, ",%f,%f,%f", mean[0], std[0], std[0]/mean[0] ); 
	meanStdDev( aspect_ratios, mean, std );
	fprintf( stdout, ",%f,%f,%f", mean[0], std[0], std[0]/mean[0] ); 
	meanStdDev( compactnesses, mean, std );
	fprintf( stdout, ",%f,%f,%f", mean[0], std[0], std[0]/mean[0] ); 
	meanStdDev( nums_holes, mean, std );
	fprintf( stdout, ",%f,%f,%f", mean[0], std[0], std[0]/mean[0] ); 
	meanStdDev( holeareas_area, mean, std );
	fprintf( stdout, ",%f,%f,%f\n", mean[0], std[0], std[0]/mean[0] ); 

	return 0;
}


