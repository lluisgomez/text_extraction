#!/bin/bash

# segmentation_mser stands for the Charles Dubout MSER implementation 
g++ -O3 -march='core2'  -c mser.cpp -o mser.o
g++ -O3 -march='core2'  -c region.cpp -o region.o
g++ -O3 -march='core2'  -c max_meaningful_clustering.cpp -o max_meaningful_clustering.o
g++ -O3 -march='core2'  -c min_bounding_box.cpp -o min_bounding_box.o
g++ -O3 -march='core2'  -c region_classifier.cpp -o region_classifier.o
g++ -O3 -march='core2'  -c group_classifier.cpp -o group_classifier.o

#g++ -O3 -march='core2' `pkg-config opencv --cflags` -c segmentation_mser.cpp -o segmentation_mser.o

#g++ -O3 -march='core2' `pkg-config opencv --cflags` -c clustering.cpp -o clustering.o

g++ -O3 -march='core2' `pkg-config opencv --cflags` -c main.cpp -o main.o 

libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o text_extraction *.o `pkg-config opencv --libs` 

# in fact not all libs are needed. code also compiles just with libopencv_core.so libopencv_features2d.so libopencv_highgui.so libopencv_imgproc.so 
