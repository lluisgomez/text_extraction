//--------------------------------------------------------------------------------------------------
// Linear time Maximally Stable Extremal Regions implementation as described in D. Nistér and
// H. Stewénius. Linear Time Maximally Stable Extremal Regions. Proceedings of the European
// Conference on Computer Vision (ECCV), 2008.
// 
// Copyright (c) 2011 Idiap Research Institute, http://www.idiap.ch/.
// Written by Charles Dubout <charles.dubout@idiap.ch>/.
// 
// MSER is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
// 
// MSER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
// 
// You should have received a copy of the GNU General Public License along with MSER. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#ifndef MSER_H
#define MSER_H

#include <vector>
#include <stdint.h>

#include "region.h"

/// The MSER class extracts maximally stable extremal regions from a grayscale (8 bits) image.
/// @note The MSER class is not reentrant, so if you want to extract regions in parallel, each
/// thread needs to have its own MSER class instance.
class MSER
{
public:
	
	/// Constructor.
	/// @param[in] eight Use 8-connected pixels instead of 4-connected.
	/// @param[in] delta DELTA parameter of the MSER algorithm. Roughly speaking, the stability of a
	/// region is the relative variation of the region area when the intensity is changed by delta.
	/// @param[in] minArea Minimum area of any stable region relative to the image domain area.
	/// @param[in] maxArea Maximum area of any stable region relative to the image domain area.
	/// @param[in] maxVariation Maximum variation (absolute stability score) of the regions.
	/// @param[in] minDiversity Minimum diversity of the regions. When the relative area of two
	/// nested regions is below this threshold, then only the most stable one is selected.
	MSER(bool eight = false, int delta = 2, double minArea = 0.0001, double maxArea = 0.5,
		 double maxVariation = 0.5, double minDiversity = 0.33);
	
	/// Extracts maximally stable extremal regions from a grayscale (8 bits) image.
	/// @param[in] bits Pointer to the first scanline of the image.
	/// @param[in] width Width of the image.
	/// @param[in] height Height of the image.
	/// @param[out] regions Detected MSER.
	void operator()(const uint8_t * bits, int width, int height, std::vector<Region> & regions);
	
private:
	// Helper method
	void processStack(int newPixelGreyLevel, int pixel, std::vector<Region *> & regionStack);
	
	// Double the size of the memory pool
	std::ptrdiff_t doublePool(std::vector<Region *> & regionStack);
	
	// Parameters
	bool eight_;
	int delta_;
	double minArea_;
	double maxArea_;
	double maxVariation_;
	double minDiversity_;
	
	// Memory pool of regions for faster allocation
	std::vector<Region> pool_;
	std::size_t poolIndex_;
};

#endif
