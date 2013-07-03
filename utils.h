
void accumulate_evidence(vector<int> *meaningful_cluster, int grow, Mat *co_occurrence)
{
	//for (int k=0; k<meaningful_clusters->size(); k++)
	   for (int i=0; i<meaningful_cluster->size(); i++)
	   	for (int j=i; j<meaningful_cluster->size(); j++)
			if (meaningful_cluster->at(i) != meaningful_cluster->at(j))
			{
			    co_occurrence->at<double>(meaningful_cluster->at(i), meaningful_cluster->at(j)) += grow;
			    co_occurrence->at<double>(meaningful_cluster->at(j), meaningful_cluster->at(i)) += grow;
			}
}

void get_gradient_magnitude(Mat& _grey_img, Mat& _gradient_magnitude)
{
	cv::Mat C = cv::Mat_<double>(_grey_img);

	cv::Mat kernel = (cv::Mat_<double>(1,3) << -1,0,1);
	cv::Mat grad_x;
	filter2D(C, grad_x, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

	cv::Mat kernel2 = (cv::Mat_<double>(3,1) << -1,0,1);
	cv::Mat grad_y;
	filter2D(C, grad_y, -1, kernel2, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

	for(int i=0; i<grad_x.rows; i++)
		for(int j=0; j<grad_x.cols; j++)
			_gradient_magnitude.at<double>(i,j) = sqrt(pow(grad_x.at<double>(i,j),2)+pow(grad_y.at<double>(i,j),2));

}

static uchar bcolors[][3] = 
{
    {0,0,255},
    {0,128,255},
    {0,255,255},
    {0,255,0},
    {255,128,0},
    {255,255,0},
    {255,0,0},
    {255,0,255},
    {255,255,255},
    {170,37,63},
    {127,86,62},
    {128,233,69},
    {203,152,120},
    {146,148,22},
    {235,218,44},
    {176,199,208},
    {144,169,24},
    {61,162,83},
    {80,242,63},
    {117,49,148},
    {203,112,62},
    {181,181,190},
    {77,46,62},
    {225,68,9},
    {188,113,155},
    {131,65,134},
    {45,90,55},
    {207,173,199},
    {193,236,5},
    {242,128,85},
    {98,190,202},
    {24,124,149},
    {70,186,74},
    {138,196,152},
    {251,95,121},
    {61,230,53},
    {151,29,185},
    {68,228,230},
    {48,233,181},
    {176,62,118},
    {110,8,104},
    {235,158,203},
    {165,232,227},
    {105,128,41},
    {201,250,179},
    {175,47,175},
    {204,232,236},
    {176,206,131},
    {154,131,199},
    {216,249,247},
    {225,98,167},
    {127,45,21},
    {103,16,0},
    {232,57,166},
    {226,236,15},
    {17,155,216},
    {250,135,135},
    {200,10,83},
    {76,209,4},
    {69,200,158},
    {167,111,118},
    {212,133,87},
    {228,133,214},
    {29,43,62},
    {10,59,38},
    {165,19,8},
    {45,155,25},
    {55,238,19},
    {9,242,220},
    {209,144,40},
    {65,7,109},
    {198,94,21},
    {75,53,233},
    {119,115,206},
    {178,153,235},
    {197,161,245},
    {96,186,155},
    {79,206,200},
    {65,170,255},
    {210,210,8},
    {217,63,218},
    {55,84,27},
    {108,62,225},
    {223,12,44},
    {120,247,163},
    {25,237,85},
    {212,136,27},
    {162,80,123},
    {76,79,202},
    {30,88,12},
    {93,50,222},
    {178,77,183},
    {240,46,238},
    {252,90,91},
    {243,254,58},
    {224,83,179},
    {104,110,204},
    {184,234,160},
    {8,180,66},
    {96,192,142},
    {146,158,172},
    {223,85,10},
    {13,68,188},
    {103,159,172},
    {101,217,168},
    {185,140,155},
    {39,89,124},
    {17,249,228},
    {198,60,157}
};

void drawClusters(Mat& img, vector<Region> *regions, vector<vector<int> > *meaningful_clusters)
{
	//img = img*0;
	uchar* rsptr = (uchar*)img.data;
	for (int i=0; i<meaningful_clusters->size(); i++)
	{

	    for (int c=0; c<meaningful_clusters->at(i).size(); c++)
	    {

		for (int p=0; p<regions->at(meaningful_clusters->at(i).at(c)).pixels_.size(); p++)
		{
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3] = bcolors[i%109][2];
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+1] = bcolors[i%109][1];
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+2] = bcolors[i%109][0];
		}
	    }
	}
}
