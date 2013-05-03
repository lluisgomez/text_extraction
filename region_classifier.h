
#ifndef REGION_CLASSIFIER_H
#define REGION_CLASSIFIER_H

#include <opencv/cv.h>
#include <opencv/ml.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "region.h"

using namespace cv;
using namespace std;

class RegionClassifier
{
public:
	
	/// Constructor.
	/// @param[in] trained_boost_filename 
	/// @param[in] decision_threshold 
	RegionClassifier(char *trained_boost_filename, float prediction_threshold=0.);
	
	/// Classify a region. Returns true iif region is classified as a text character
	/// @param[in] regions A pointer to the region to be classified.
	bool operator()(Region *region);
	
	/// Classify a region. Returns the average classification votes
	/// @param[in] regions A pointer to the region to be classified.
	float get_votes(Region *region);

private:
	
	// Boosted tree classifier
	CvBoost boost_;
		
	// Classification parameter
	float decision_threshold_;
};

#endif
