#include <cstdlib>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

/* STEP 2. Opening the file */
//1. Declare a structure to keep the data
CvMLData cvml;

//2. Read the file
cvml.read_csv("groups_dataset.csv");
//cvml.read_csv("strokes_dataset_noresized.csv");

//3. Indicate which column is the response
cvml.set_response_idx(0);


/* STEP 3. Splitting the samples */
//1. Select 50% for the training (an integer value is also allowed here)
CvTrainTestSplit cvtts(0.9f, true);
//2. Assign the division to the data
cvml.set_train_test_split(&cvtts);

/* STEP 4. The training */
//1. Declare the classifier
CvBoost boost;

ifstream ifile("./trained_boost_groups.xml");
if (ifile) 
{
	// The file exists, so we don't need to train 
	boost.load("./trained_boost_groups.xml", "boost");
} else {
	//2. Train it with 100 features
	printf("Training ... \n");
	boost.train(&cvml, CvBoostParams(CvBoost::REAL, 500, 0, 1, false, 0), false);
}

/* STEP 5. Calculating the testing and training error */
// 1. Declare a couple of vectors to save the predictions of each sample
std::vector<float> train_responses, test_responses;
// 2. Calculate the training error
float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR,&train_responses);
// 3. Calculate the test error
float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR,&test_responses);
printf("Error train %f \n", fl1);
printf("Error test %f \n", fl2);

static const float arr[] = {0,-1.980394,1.249858,-0.631116,2.819193,0.305448,0.108346,0.801116,0.104873,0.130908,0.559806,0.255053,0.455610,0.294118,0.455645,1.549193,0.087770,0.144896,1.650866};
vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );
float prediction = boost.predict( Mat(sample), Mat(), Range::all(), false, false );
float votes      = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

printf("\n The group sample is predicted as: %f (with number of votes = %f)\n", prediction,votes);

//static const float arr2[] = {0,0.911369,1.052156,1.154478,3.321924,0.829768,0.249785,0.616930,0.246637,0.399782,0.337159,0.103893,0.308142,0.666667,0.745356,1.118034,0.009747,0.011016,1.130162};
static const float arr2[] = {0,1.14335,3.00412,2.62747,3.26428,2.32749,0.713018,0.47244,0.289846,0.613508,0.40514,0.216716,0.53305,0.878788,3.21698,3.6607,0.0422318,0.114392,2.70868};
vector<float> sample2 (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );
float prediction2 = boost.predict( Mat(sample2), Mat(), Range::all(), false, false );
float votes2      = boost.predict( Mat(sample2), Mat(), Range::all(), false, true );

printf("\n The group sample is predicted as: %f (with number of votes = %f)\n", prediction2,votes2);

/* STEP 6. Save your classifier */
// Save the trained classifier
boost.save("./trained_boost_groups.xml", "boost");

return EXIT_SUCCESS;
}
