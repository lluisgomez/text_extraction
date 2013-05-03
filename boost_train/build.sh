#!/bin/bash

g++ -O3 -march='core2' `pkg-config opencv --cflags` -c extract_char_features.cpp -o extract_char_features.o 
libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o extract_char_features extract_char_features.o `pkg-config opencv --libs` 

g++ -O3 -march='core2' `pkg-config opencv --cflags` -c boost_char_train.cpp -o boost_char_train.o 
libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o boost_char_train boost_char_train.o `pkg-config opencv --libs` 

g++ -O3 -march='core2' `pkg-config opencv --cflags` -c extract_group_features.cpp -o extract_group_features.o 
libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o extract_group_features extract_group_features.o `pkg-config opencv --libs` 

g++ -O3 -march='core2' `pkg-config opencv --cflags` -c boost_groups_train.cpp -o boost_groups_train.o 
libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o boost_groups_train boost_groups_train.o `pkg-config opencv --libs` 
