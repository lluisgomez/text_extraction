
#ifndef GROUP_CLASSIFIER_H
#define GROUP_CLASSIFIER_H

#include <opencv/cv.h>
#include <opencv/ml.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "region.h"
#include "region_classifier.h"

using namespace cv;
using namespace std;

class GroupClassifier
{
public:
	
	/// Constructor.
	/// @param[in] trained_boost_filename 
	/// @param[in] character_classifier a pointer to CvBoost for character classification
	GroupClassifier(char *trained_boost_filename, RegionClassifier *character_classifier);
	
	/// Classify a region. Returns true iif a group of regions is classified as a text group
	/// @param[in] regions A pointer to the group of regions indexes to be classified.
	/// @param[in] regions A pointer to the whole regions vector.
	double operator()(vector<int> *group, vector<Region> *regions);
	

private:
	
	// Boosted tree classifier
	CvBoost boost_;
	
	// Boosted tree classifier for single characters
	RegionClassifier *character_classifier_;
		
	// Classification parameter
	float decision_threshold_;
};

#endif
