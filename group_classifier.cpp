#include "group_classifier.h"

GroupClassifier::GroupClassifier(char *trained_boost_filename, RegionClassifier *character_classifier) :  character_classifier_(character_classifier)
{

	assert(trained_boost_filename != NULL);
	assert(character_classifier != NULL);

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

double GroupClassifier::operator()(vector<int> *group, vector<Region> *regions)
{
	assert(group != NULL);
	assert(group->size() > 1);
	assert(regions != NULL);

	Mat votes          ( group->size(), 1, CV_32F, 1 );
	Mat strokes     ( group->size(), 1, CV_32F, 1 );
	Mat aspect_ratios  ( group->size(), 1, CV_32F, 1 );
	Mat compactnesses  ( group->size(), 1, CV_32F, 1 );
	Mat nums_holes     ( group->size(), 1, CV_32F, 1 );
	Mat holeareas_area ( group->size(), 1, CV_32F, 1 );

	for (int i=group->size()-1; i>=0; i--)
	{
		// TODO check first if regions->at(group->at(i)).votes_ has already been calculated !!! 
		regions->at(group->at(i)).classifier_votes_ = character_classifier_->get_votes(&regions->at(group->at(i)));

		votes.at<float>(i,0) = regions->at(group->at(i)).classifier_votes_;
		strokes.at<float>(i,0) = (float)regions->at(group->at(i)).stroke_mean_;
		aspect_ratios.at<float>(i,0) = (float)min( regions->at(group->at(i)).rect_.size.width, regions->at(group->at(i)).rect_.size.height)/max( regions->at(group->at(i)).rect_.size.width, regions->at(group->at(i)).rect_.size.height);
		compactnesses.at<float>(i,0) = sqrt(regions->at(group->at(i)).area_)/regions->at(group->at(i)).perimeter_;
		nums_holes.at<float>(i,0) = (float)regions->at(group->at(i)).num_holes_;
		holeareas_area.at<float>(i,0) = (float)regions->at(group->at(i)).holes_area_/regions->at(group->at(i)).area_;
	}

        vector<float> sample;
	sample.push_back(0);
	
	Scalar mean,std;
	meanStdDev( votes, mean, std );
	sample.push_back( mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 
	meanStdDev( strokes, mean, std );
	sample.push_back( mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 
	meanStdDev( aspect_ratios, mean, std );
	sample.push_back( mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 
	meanStdDev( compactnesses, mean, std );
	sample.push_back( mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 
	meanStdDev( nums_holes, mean, std );
	sample.push_back( mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 
	meanStdDev( holeareas_area, mean, std );
	sample.push_back(mean[0]);
	sample.push_back( std[0]);
	sample.push_back( std[0]/mean[0] ); 


	float votes_group = boost_.predict( Mat(sample), Mat(), Range::all(), false, true );

  return (double)1-(double)1/(1+exp(-2*votes_group));
}
