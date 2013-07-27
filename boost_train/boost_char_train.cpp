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
cvml.read_csv("char_dataset.csv");
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

ifstream ifile("./trained_boost_char.xml");
if (ifile) 
{
	// The file exists, so we don't need to train 
	boost.load("./trained_boost_char.xml", "boost");
} else {
	//2. Train it with 100 features
	printf("Training ... \n");
	boost.train(&cvml, CvBoostParams(CvBoost::REAL, 200, 0, 1, false, 0), false);
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


//Try a char
static const float arr[] = {0,1.659899,0.684169,0.412175,150.000000,81.000000,0.540000,0.358025,0.151203,0.000000,0.000000};

vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );
float prediction = boost.predict( Mat(sample), Mat(), Range::all(), false, false );
float votes      = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

printf("\n The sample (360) is predicted as: %f (with number of votes = %f)\n", prediction,votes);

//Try a NONchar
static const float arr2[] = {0,1.250000,0.433013,0.346410,9.000000,8.000000,0.888889,0.833333,0.375000,0.000000,0.000000};

vector<float> sample2 (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );
prediction = boost.predict( Mat(sample2), Mat(), Range::all(), false, false );
votes      = boost.predict( Mat(sample2), Mat(), Range::all(), false, true );

printf("\n The sample (367) is predicted as: %f (with number of votes = %f)\n", prediction,votes);

/* STEP 6. Save your classifier */
// Save the trained classifier
boost.save("./trained_boost_char.xml", "boost");

return EXIT_SUCCESS;
}
