#define _MAIN

#include <opencv/highgui.h>

#include <iostream>
#include <fstream>
#include <stdio.h>

#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"

#define NUM_FEATURES 11 

#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

using namespace std;
using namespace cv;

#include "utils.h"

int main( int argc, char** argv )
{

	
    Mat img, grey, lab_img, gradient_magnitude, segmentation, all_segmentations;

    vector<Region> regions;
    ::MSER mser8(false,25,0.000008,0.03,1,0.7);

    RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0); 
    GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost); 

    img = imread(argv[1]);
    cvtColor(img, grey, CV_BGR2GRAY);
    cvtColor(img, lab_img, CV_BGR2Lab);
    gradient_magnitude = Mat_<double>(img.size());
    get_gradient_magnitude( grey, gradient_magnitude);

		segmentation = Mat::zeros(img.size(),CV_8UC3);
		all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
	
    for (int step =1; step<3; step++)
    {


	  if (step == 2)
		  grey = 255-grey;

	  //double t_tot = (double)cvGetTickCount();

    //double t = (double)cvGetTickCount();
    mser8((uchar*)grey.data, grey.cols, grey.rows, regions);


    //t = cvGetTickCount() - t;
    //cout << "Detected " << regions.size() << " regions" << " in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    for (int i=0; i<regions.size(); i++)
      regions[i].er_fill(grey);

    //t = cvGetTickCount() - t;
    //cout << "Regions filled in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();


    double max_stroke = 0;
    for (int i=regions.size()-1; i>=0; i--)
    {
      regions[i].extract_features(lab_img, grey, gradient_magnitude);
      if ( (regions.at(i).stroke_std_/regions.at(i).stroke_mean_ > 0.8) || (regions.at(i).num_holes_>2) || (regions.at(i).bbox_.width <=3) || (regions.at(i).bbox_.height <=3) )
      	regions.erase(regions.begin()+i);
      else 
        max_stroke = max(max_stroke, regions[i].stroke_mean_);
    }

    //t = cvGetTickCount() - t;
    //cout << "Features extracted in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);

    vector< vector<int> > meaningful_clusters;
    vector< vector<int> > final_clusters;
    Mat co_occurrence_matrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);

    int dims[NUM_FEATURES] = {3,3,3,3,3,3,3,3,3,5,5};

    for (int f=0; f<NUM_FEATURES; f++)
    {
      unsigned int N = regions.size();
      if (N<3) break;
      int dim = dims[f];
      t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));
      int count = 0;
      for (int i=0; i<regions.size(); i++)
      {
        data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/img.cols;
        data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/img.rows;
        switch (f)
        {
          case 0:
              data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
              break;	
          case 1:
              data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;	
              break;	
          case 2:
              data[count+2] = (t_float)regions.at(i).bbox_.y/img.rows;	
              break;	
          case 3:
              data[count+2] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height)/img.rows;	
              break;	
          case 4:
              data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(img.rows,img.cols);	
              break;	
          case 5:
              data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;	
              break;	
          case 6:
              data[count+2] = (t_float)regions.at(i).area_/(img.rows*img.cols);	
              break;	
          case 7:
              data[count+2] = (t_float)(regions.at(i).bbox_.height*regions.at(i).bbox_.width)/(img.rows*img.cols);	
              break;	
          case 8:
              data[count+2] = (t_float)regions.at(i).gradient_mean_/255;	
              break;	
          case 9:
              data[count+2] = (t_float)regions.at(i).color_mean_.at(0)/255;	
              data[count+3] = (t_float)regions.at(i).color_mean_.at(1)/255;	
              data[count+4] = (t_float)regions.at(i).color_mean_.at(2)/255;	
              break;	
          case 10:
              data[count+2] = (t_float)regions.at(i).boundary_color_mean_.at(0)/255;	
              data[count+3] = (t_float)regions.at(i).boundary_color_mean_.at(1)/255;	
              data[count+4] = (t_float)regions.at(i).boundary_color_mean_.at(2)/255;	
              break;	
        }
        count = count+dim;
      }

      mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters); // TODO try accumulating more evidence by using different methods and metrics

      for (int k=0; k<meaningful_clusters.size(); k++)
      {
          //if ( group_boost(&meaningful_clusters.at(k), &regions)) // TODO try is it's betetr to accumulate only the most probable text groups
        accumulate_evidence(&meaningful_clusters.at(k), 1, &co_occurrence_matrix);

        if ( (group_boost(&meaningful_clusters.at(k), &regions) >= DECISION_THRESHOLD_SF) )
        {
          final_clusters.push_back(meaningful_clusters.at(k));
        }
      }
      
      Mat tmp_segmentation = Mat::zeros(img.size(),CV_8UC3);
      Mat tmp_all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
      drawClusters(tmp_segmentation, &regions, &meaningful_clusters);
      Mat tmp = Mat::zeros(240,320,CV_8UC3);
      resize(tmp_segmentation,tmp,tmp.size());
      tmp.copyTo(tmp_all_segmentations(Rect(320*f,0,320,240)));
      all_segmentations = all_segmentations + tmp_all_segmentations;

      free(data);
      meaningful_clusters.clear();
    }
    //t = cvGetTickCount() - t;
    //cout << "Clusterings (" << NUM_FEATURES << ") done in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
    //t = (double)cvGetTickCount();

    /**/
    double minVal;
    double maxVal;
    minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);

    maxVal = NUM_FEATURES - 1; //TODO this is true only if you are using "grow == 1" in accumulate_evidence function
    minVal=0;

    co_occurrence_matrix = maxVal - co_occurrence_matrix;
    co_occurrence_matrix = co_occurrence_matrix / maxVal;

    //we want a sparse matrix
    
    t_float *D = (t_float*)malloc((regions.size()*regions.size()) * sizeof(t_float)); 
    int pos = 0;
    for (int i = 0; i<co_occurrence_matrix.rows; i++)
    {
      for (int j = i+1; j<co_occurrence_matrix.cols; j++)
      {
        D[pos] = (t_float)co_occurrence_matrix.at<double>(i, j);
        pos++;
      }
    }
    
    // fast clustering from the co-occurrence matrix
    mm_clustering(D, regions.size(), METHOD_METR_AVERAGE, &meaningful_clusters); //  TODO try with METHOD_METR_COMPLETE
    free(D);
    
    //t = cvGetTickCount() - t;
    //cout << "Evidence Accumulation Clustering done in " << t/((double)cvGetTickFrequency()*1000.) << " ms. Got " << meaningful_clusters.size() << " clusters." << endl;
    //t = (double)cvGetTickCount();


    for (int i=meaningful_clusters.size()-1; i>=0; i--)
    {
      //if ( (! group_boost(&meaningful_clusters.at(i), &regions)) || (meaningful_clusters.at(i).size()<3) )
      if ( (group_boost(&meaningful_clusters.at(i), &regions) >= DECISION_THRESHOLD_EA) )
      {
      	final_clusters.push_back(meaningful_clusters.at(i));
      }
    }

    drawClusters(segmentation, &regions, &final_clusters);

    if (step == 2)
    {	
      cvtColor(segmentation, grey, CV_BGR2GRAY);
      threshold(grey,grey,1,255,CV_THRESH_BINARY);
      imwrite("out.png", grey);
      
      if (argc > 2)
      {
        Mat gt;
        gt = imread(argv[2]);
        cvtColor(gt, gt, CV_RGB2GRAY);
        threshold(gt, gt, 1, 255, CV_THRESH_BINARY_INV); // <- for KAIST gt
        //threshold(gt, gt, 254, 255, CV_THRESH_BINARY); // <- for ICDAR gt
        Mat tmp_mask = (255-gt) & (grey);
        cout << "Pixel level recall = " << (float)countNonZero(tmp_mask) / countNonZero(255-gt) << endl;
        cout << "Pixel level precission = " << (float)countNonZero(tmp_mask) / countNonZero(grey) << endl;
      }
      else
      {
        imshow("Original", img);
        imshow("Text extraction", segmentation);
        waitKey(0);
      }

    }


    regions.clear();
    //t_tot = cvGetTickCount() - t_tot;
    //cout << " Total processing for one frame " << t_tot/((double)cvGetTickFrequency()*1000.) << " ms." << endl;

    }

}
