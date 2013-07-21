#include "region_classifier.h"

RegionClassifier::RegionClassifier(char *trained_boost_filename, float decision_threshold) :  decision_threshold_(decision_threshold)
{

	assert(trained_boost_filename != NULL);

	ifstream ifile1(trained_boost_filename);
	if (ifile1) 
	{
		//fprintf(stdout,"Loading boost character classifier ... \n");
		boost_.load(trained_boost_filename, "boost");
	} else {
		fprintf(stderr,"Boost character classifier, file not found! \n");
		exit(-1);
	}
}

bool RegionClassifier::operator()(Region *region)
{
	assert(region != NULL);

	float sample_arr[] = {0, region->stroke_mean_, region->stroke_std_, region->stroke_std_/region->stroke_mean_, (float)region->area_, (float)region->perimeter_, (float)region->perimeter_/region->area_, (float)min( region->rect_.size.width, region->rect_.size.height)/max( region->rect_.size.width, region->rect_.size.height), sqrt(region->area_)/region->perimeter_, (float)region->num_holes_, (float)region->holes_area_/region->area_};
        vector<float> sample (sample_arr, sample_arr + sizeof(sample_arr) / sizeof(sample_arr[0]) );

	float votes = boost_.predict( Mat(sample), Mat(), Range::all(), false, true );

	if (votes <= decision_threshold_)
		return true;

	return false;
}

float RegionClassifier::get_votes(Region *region)
{
	assert(region != NULL);

	float sample_arr[] = {0, region->stroke_mean_, region->stroke_std_, region->stroke_std_/region->stroke_mean_, (float)region->area_, (float)region->perimeter_, (float)region->perimeter_/region->area_, (float)min( region->rect_.size.width, region->rect_.size.height)/max( region->rect_.size.width, region->rect_.size.height), sqrt(region->area_)/region->perimeter_, (float)region->num_holes_, (float)region->holes_area_/region->area_};
        vector<float> sample (sample_arr, sample_arr + sizeof(sample_arr) / sizeof(sample_arr[0]) );

	return boost_.predict( Mat(sample), Mat(), Range::all(), false, true );
}
