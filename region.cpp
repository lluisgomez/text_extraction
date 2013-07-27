

#include "region.h"

#include <algorithm>
#include <cassert>
#include <limits>

using namespace std;
using namespace cv;

Region::Region(int level, int pixel) : level_(level), pixel_(pixel), area_(0),
variation_(numeric_limits<double>::infinity()), stable_(false), parent_(0), child_(0), next_(0), bbox_x1_(10000), bbox_y1_(10000), bbox_x2_(0), bbox_y2_(0)
{
	fill_n(moments_, 5, 0.0);
}

//inline void Region::accumulate(int x, int y)
void Region::accumulate(int x, int y)
{
	++area_;
	moments_[0] += x;
	moments_[1] += y;
	moments_[2] += x * x;
	moments_[3] += x * y;
	moments_[4] += y * y;

	bbox_x1_ = min(bbox_x1_, x);
	bbox_y1_ = min(bbox_y1_, y);
	bbox_x2_ = max(bbox_x2_, x);
	bbox_y2_ = max(bbox_y2_, y);
}

void Region::merge(Region * child)
{
	assert(!child->parent_);
	assert(!child->next_);
	
	// Add the moments together
	area_ += child->area_;
	moments_[0] += child->moments_[0];
	moments_[1] += child->moments_[1];
	moments_[2] += child->moments_[2];
	moments_[3] += child->moments_[3];
	moments_[4] += child->moments_[4];

	// Rebuild bounding box 
	bbox_x1_ = min(bbox_x1_, child->bbox_x1_);
	bbox_y1_ = min(bbox_y1_, child->bbox_y1_);
	bbox_x2_ = max(bbox_x2_, child->bbox_x2_);
	bbox_y2_ = max(bbox_y2_, child->bbox_y2_);
	
	child->next_ = child_;
	child_ = child;
	child->parent_ = this;
}

void Region::process(int delta, int minArea, int maxArea, double maxVariation)
{
	// Find the last parent with level not higher than level + delta
	const Region * parent = this;
	
	while (parent->parent_ && (parent->parent_->level_ <= (level_ + delta)))
		parent = parent->parent_;
	
	// Calculate variation
	variation_ = static_cast<double>(parent->area_ - area_) / area_;
	
	// Whether or not the region *could* be stable
	const bool stable = (!parent_ || (variation_ <= parent_->variation_)) &&
						(area_ >= minArea) && (area_ <= maxArea) && (variation_ <= maxVariation);
	
	// Process all the children
	for (Region * child = child_; child; child = child->next_) {
		child->process(delta, minArea, maxArea, maxVariation);
		
		if (stable && (variation_ < child->variation_))
			stable_ = true;
	}
	
	// The region can be stable even without any children
	if (!child_ && stable)
		stable_ = true;
}

bool Region::check(double variation, int area) const
{
	if (area_ <= area)
		return true;
	
	if (stable_ && (variation_ < variation))
		return false;
	
	for (Region * child = child_; child; child = child->next_)
		if (!child->check(variation, area))
			return false;
	
	return true;
}

void Region::save(double minDiversity, vector<Region> & regions)
{
	if (stable_) {
		const int minParentArea = area_ / (1.0 - minDiversity) + 0.5;
		
		const Region * parent = this;
		
		while (parent->parent_ && (parent->parent_->area_ < minParentArea)) {
			parent = parent->parent_;
			
			if (parent->stable_ && (parent->variation_ <= variation_)) {
				stable_ = false;
				break;
			}
		}
		
		if (stable_) {
			const int maxChildArea = area_ * (1.0 - minDiversity) + 0.5;
			
			if (!check(variation_, maxChildArea))
				stable_ = false;
		}
		
		if (stable_) {
			regions.push_back(*this);
			regions.back().parent_ = 0;
			regions.back().child_ = 0;
			regions.back().next_ = 0;
		}
	}
	
	for (Region * child = child_; child; child = child->next_)
		child->save(minDiversity, regions);
}

void Region::detect(int delta, int minArea, int maxArea, double maxVariation,
						  double minDiversity, vector<Region> & regions)
{
	process(delta, minArea, maxArea, maxVariation);
	save(minDiversity, regions);
}

/* function:    er_fill is borowed from vlfeat-0.9.14/toolbox/mser/vl_erfill.c
** description: Extremal Regions filling
** author:      Andrea Vedaldi
**/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

The function is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/
void Region::er_fill(Mat& _grey_img)
{
	const uint8_t *src = (uint8_t*)_grey_img.data;
	

	double er = pixel_; 
	int ndims = 2;
    	int dims [2];
	dims[0] = _grey_img.cols;
	dims[1] = _grey_img.rows;
  	int last = 0 ;
  	int last_expanded = 0 ;
  	uint8_t value = 0 ;

	double const * er_pt ;

  	int*   subs_pt ;       /* N-dimensional subscript                 */
  	int*   nsubs_pt ;      /* diff-subscript to point to neigh.       */
  	int*   strides_pt ;    /* strides to move in image array          */
  	uint8_t*  visited_pt ;    /* flag                                    */
  	int*   members_pt ;    /* region members                          */
  	bool   invert = false;
	
	/* get dimensions */
  	int nel   = dims[0]*dims[1];
  	uint8_t *I_pt  = (uint8_t *)src;

	/* allocate stuff */
	subs_pt    = (int*) malloc( sizeof(int)      * ndims ) ;
	nsubs_pt   = (int*) malloc( sizeof(int)      * ndims ) ;
	strides_pt = (int*) malloc( sizeof(int)      * ndims ) ;
	visited_pt = (uint8_t*)malloc( sizeof(uint8_t)     * nel   ) ;
	members_pt = (int*) malloc( sizeof(int)      * nel   ) ;

	er_pt = &er;

	/* compute strides to move into the N-dimensional image array */
	strides_pt [0] = 1 ;
	int k;
	for(k = 1 ; k < ndims ; ++k) {
	  strides_pt [k] = strides_pt [k-1] * dims [k-1] ;
	}

	//fprintf(stderr,"strides_pt %d %d \n",strides_pt [0],strides_pt [1]);

	/* load first pixel */
	memset(visited_pt, 0, sizeof(uint8_t) * nel) ;
	{
	  int idx = (int) *er_pt ;
	  if (idx < 0) {
	    idx = -idx;
	    invert = true ;
	  }
	  if( idx < 0 || idx > nel+1 ) {
	    fprintf(stderr,"ER=%d out of range [1,%d]",idx,nel) ;
	    return;
	  }
	  members_pt [last++] = idx ;
	}
	value = I_pt[ members_pt[0] ]  ;

	/* -----------------------------------------------------------------
	 *                                                       Fill region
	 * -------------------------------------------------------------- */
	while(last_expanded < last) {

	  /* pop next node xi */
	  int index = members_pt[last_expanded++] ;

	  /* convert index into a subscript sub; also initialize nsubs
	     to (-1,-1,...,-1) */
	  {
	    int temp = index ;
	    for(k = ndims-1 ; k >=0 ; --k) {
	      nsubs_pt [k] = -1 ;
	      subs_pt  [k] = temp / strides_pt [k] ;
	      temp         = temp % strides_pt [k] ;
	    }
	  }

	  /* process neighbors of xi */
	  while(true) {
	    int good = true ;
	    int nindex = 0 ;

	    /* compute NSUBS+SUB, the correspoinding neighbor index NINDEX
	       and check that the pixel is within image boundaries. */
	    for(k = 0 ; k < ndims && good ; ++k) {
	      int temp = nsubs_pt [k] + subs_pt [k] ;
	      good &= 0 <= temp && temp < (signed) dims[k] ;
	      nindex += temp * strides_pt [k] ;
	    }

	    /* process neighbor
	       1 - the pixel is within image boundaries;
	       2 - the pixel is indeed different from the current node
	       (this happens when nsub=(0,0,...,0));
	       3 - the pixel has value not greather than val
	       is a pixel older than xi
	       4 - the pixel has not been visited yet
	    */
	    if(good
	       && nindex != index
	       && ((!invert && I_pt [nindex] <= value) ||
	           ( invert && I_pt [nindex] >= value))
	       && ! visited_pt [nindex] ) {

		//fprintf(stderr,"nvalue %d  value %d",(int)(I_pt [nindex]),(int)(I_pt [index]));
	    	//fprintf(stderr,"           index %d\n",index);
	    	//fprintf(stderr,"neightbour index %d\n",nindex);

	      /* mark as visited */
	      visited_pt [nindex] = 1 ;

	      /* add to list */
	      members_pt [last++] = nindex ;
	    }

	    /* move to next neighbor */
	    k = 0 ;
	    while(++ nsubs_pt [k] > 1) {
	      nsubs_pt [k++] = -1 ;
	      if(k == ndims) goto done_all_neighbors ;
	    }
	  } /* next neighbor */
	done_all_neighbors : ;
	} /* goto pop next member */

	/*
	 * Save results
	 */
	{
	  for (int i = 0 ; i < last ; ++i) {
	    pixels_.push_back(members_pt[i]);
	    //fprintf(stderr,"	pixel inserted %d: %d\n",i,members_pt[i]);
	  }
	}
	
	
	free( members_pt ) ;
	free( visited_pt ) ;
	free( strides_pt ) ;
	free( nsubs_pt   ) ;
	free( subs_pt    ) ;
	
	return;
}

void Region::extract_features(Mat& _lab_img, Mat& _grey_img, Mat& _gradient_magnitude)
{

	bbox_x2_++;
	bbox_y2_++;

	center_.x = bbox_x2_-bbox_x1_ / 2;
	center_.y = bbox_y2_-bbox_y1_ / 2;

	bbox_ = cvRect(bbox_x1_,bbox_y1_,bbox_x2_-bbox_x1_,bbox_y2_-bbox_y1_);	

	Mat canvas = Mat::zeros(_lab_img.size(),CV_8UC1);
	uchar* rsptr = (uchar*)canvas.data;
	for (int p=0; p<pixels_.size(); p++)
	{
		rsptr[pixels_[p]] = 255;
	}

	int x = bbox_x1_ - min (10, bbox_x1_);
	int y = bbox_y1_ - min (10, bbox_y1_);
	int width  = bbox_x2_ - x + min(10, _lab_img.cols-bbox_x2_);
	int height = bbox_y2_ - y + min(10, _lab_img.rows-bbox_y2_);

	CvRect rect = cvRect(x, y, width, height);
	Mat bw = canvas(rect);
	
	Scalar mean,std;
	meanStdDev(_grey_img(rect),mean,std,bw);
	intensity_mean_ = mean[0];
	intensity_std_  = std[0];
	
	meanStdDev(_lab_img(rect),mean,std,bw);
	color_mean_.push_back(mean[0]);
	color_mean_.push_back(mean[1]);
	color_mean_.push_back(mean[2]);
	color_std_.push_back(std[0]);
	color_std_.push_back(std[1]);
	color_std_.push_back(std[2]);

	Mat tmp;
	distanceTransform(bw, tmp, CV_DIST_L1,3); //L1 gives distance in round integers while L2 floats

        meanStdDev(tmp,mean,std,bw);
	stroke_mean_ = mean[0];
	stroke_std_  = std[0];

	Mat element = getStructuringElement( MORPH_RECT, Size(5, 5), Point(2, 2) );
	dilate(bw, tmp, element);
	absdiff(tmp, bw, tmp);	
	
	meanStdDev(_grey_img(rect), mean, std, tmp);
	boundary_intensity_mean_ = mean[0];
	boundary_intensity_std_  = std[0];
	
	meanStdDev(_lab_img(rect), mean, std, tmp);
	boundary_color_mean_.push_back(mean[0]);
	boundary_color_mean_.push_back(mean[1]);
	boundary_color_mean_.push_back(mean[2]);
	boundary_color_std_.push_back(std[0]);
	boundary_color_std_.push_back(std[1]);
	boundary_color_std_.push_back(std[2]);

	Mat tmp2;
	dilate(bw, tmp, element);
	erode(bw, tmp2, element);
	absdiff(tmp, tmp2, tmp);	
	
	meanStdDev(_gradient_magnitude(rect), mean, std, tmp);
	gradient_mean_ = mean[0];
	gradient_std_  = std[0];


	copyMakeBorder(bw, bw, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));

	num_holes_ = 0;
	holes_area_ = 0;
	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	findContours( bw, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    	for (int k=0; k<hierarchy.size();k++)
    	{
	    //TODO check this thresholds, are they coherent? is there a faster way?
	    if ((hierarchy[k][3]==0)&&((((float)contourArea(Mat(contours0.at(k)))/contourArea(Mat(contours0.at(0))))>0.01)||(contourArea(Mat(contours0.at(k)))>31)))
	    {
		num_holes_++;
		holes_area_ += (int)contourArea(Mat(contours0.at(k)));
	    }
    	}
	perimeter_ = (int)contours0.at(0).size();
	rect_ = minAreaRect(contours0.at(0));
}
