
#ifndef REGION_H
#define REGION_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <stdint.h>

/// A Maximally Stable Extremal Region.
class Region
{
public:
	int level_; ///< Level at which the region is processed.
	int pixel_; ///< Index of the initial pixel (y * width + x).
	int area_; ///< Area of the region (moment zero).
	double moments_[5]; ///< First and second moments of the region (x, y, x^2, xy, y^2).
	double variation_; ///< MSER variation.

	/// Axis oriented bounding box of the region
        int bbox_x1_;
        int bbox_y1_;
        int bbox_x2_;
        int bbox_y2_;

	/// Constructor.
	/// @param[in] level Level at which the region is processed.
	/// @param[in] pixel Index of the initial pixel (y * width + x).
	Region(int level = 256, int pixel = 0);

	/// Fills an Extremal Region (ER) by region growing from the Index of the initial pixel(pixel_).
	/// @param[in] grey_img Grey level image
	void er_fill(cv::Mat& _grey_img);

	std::vector<int> pixels_; ///< list pf all pixels indexes (y * width + x) of the region

	/// Extract_features.
	/// @param[in] lab_img L*a*b* color image to extract color information
	/// @param[in] grey_img Grey level version of the original image 
	/// @param[in] gradient_magnitude of the original image
	void extract_features(cv::Mat& _lab_img, cv::Mat& _grey_img, cv::Mat& _gradient_magnitude);

	cv::Point center_;	///< Center coordinates of the region
	cv::Rect bbox_;		///< Axis aligned bounding box
	cv::RotatedRect rect_;		///< Axis aligned bounding box
	int perimeter_;		///< Perimeter of the region
	int num_holes_;		///< Number of holes of the region
	int holes_area_;	///< Total area filled by all holes of this regions
        float intensity_mean_;	///< mean intensity of the whole region
        float intensity_std_;	///< intensity standard deviation of the whole region
	std::vector<float> color_mean_;	///< mean color (L*a*b*)  of the whole region
	std::vector<float> color_std_;	///< color (L*a*b*) standard deviation of the whole region
        float boundary_intensity_mean_;	///< mean intensity of the boundary of the region
        float boundary_intensity_std_;	///< intensity standard deviation of the boundary of the region
	std::vector<float> boundary_color_mean_; ///< mean color (L*a*b*)  of the boundary of the region
	std::vector<float> boundary_color_std_;	 ///< color (L*a*b*) standard deviation of the boundary of the region
	double stroke_mean_;	///< mean stroke of the whole region
	double stroke_std_;	///< stroke standard deviation of the whole region
	double gradient_mean_;	///< mean gradient magnitude of the whole region
	double gradient_std_;	///< gradient magnitude standard deviation of the whole region

	float classifier_votes_; ///< Votes of the Region_Classifier for this region
	
private:
	bool stable_; // Flag indicating if the region is stable
	Region * parent_; // Pointer to the parent region
	Region * child_; // Pointer to the first child
	Region * next_; // Pointer to the next (sister) region
	
	void accumulate(int x, int y);
	void merge(Region * child);
	void detect(int delta, int minArea, int maxArea, double maxVariation, double minDiversity,
				std::vector<Region> & regions);
	void process(int delta, int minArea, int maxArea, double maxVariation);
	bool check(double variation, int area) const;
	void save(double minDiversity, std::vector<Region> & regions);
	
	friend class MSER;
};

#endif
