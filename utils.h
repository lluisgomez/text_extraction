
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
    {255,255,255}
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
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3] = bcolors[i%9][2];
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+1] = bcolors[i%9][1];
			rsptr[regions->at(meaningful_clusters->at(i).at(c)).pixels_.at(p)*3+2] = bcolors[i%9][0];
		}
	    }
	}
}
