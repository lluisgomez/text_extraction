text_extraction
===============

This code is the implementation of the method proposed in the paper “Multi-script text extraction from natural scenes” (Gomez &amp; Karatzas), International Conference on Document Analysis and Recognition, ICDAR2013.

This code should reproduce the same quantitative results published on the paper for the KAIST dataset (for the task of text segmentation at pixel level). If you plan to compare this method with your's in other datasets please drop us a line ({lgomez,dimos}@cvc.uab.es). Thanks!


Includes the following third party code:

  - fast_clustering.cpp Copyright (c) 2011 Daniel Müllner, under the BSD license. http://math.stanford.edu/~muellner/fastcluster.html
  - mser.cpp Copyright (c) 2011 Idiap Research Institute, under the GPL license. http://www.idiap.ch/~cdubout/
  - binomial coefficient approximations are due to Rafael Grompone von Gioi. http://www.ipol.im/pub/art/2012/gjmr-lsd/
