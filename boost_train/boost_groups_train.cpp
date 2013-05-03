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
CvTrainTestSplit cvtts(0.8f, true);
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

static const float arr[] = {0.000000,1.251828,0.651314,0.520290,0.116600,0.164898,1.414214,0.316165,0.232621,0.735758,0.280476,0.124072,0.442363,0.333333,0.471405,1.414214,0.003945,0.005579,1.414214};
vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );
float prediction = boost.predict( Mat(sample), Mat(), Range::all(), false, false );
float votes      = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

printf("\n The sample (360) is predicted as: %f (with number of votes = %f)\n", prediction,votes);

/* STEP 6. Save your classifier */
// Save the trained classifier
boost.save("./trained_boost_groups.xml", "boost");

return EXIT_SUCCESS;
}
