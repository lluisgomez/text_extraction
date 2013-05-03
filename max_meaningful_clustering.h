
#ifndef MAX_MEANINGFUL_CLUSTERING_H
#define MAX_MEANINGFUL_CLUSTERING_H

#include <vector>

#include "fast_clustering.cpp"
#include "nfa.cpp"
#include "min_bounding_box.h"

using namespace std;

typedef struct {
        int num_elem; 		// number of elements
        vector<int> elements;   // elements (contour ID)
	int nfa;		// the number of false alarms for this merge (we are using only the nfa exponent so this is an int)
        float dist;		// distance of the merge		
        float dist_ext;		// distamce where this merge will merge with another
        long double volume;		// volume of the bounding sphere (or bounding box)
        long double volume_ext;		// volume of the sphere(or box) + envolvent empty space
        vector<vector<float> > points; // nD points in this cluster
	bool   max_meaningful;	 //is this merge max meaningul ?
	vector<int>    max_in_branch;	 //otherwise which merges are the max_meaningful in this branch
	int min_nfa_in_branch;//here we store the min nfa detected within the chilhood of this merge and this one (we are using only the nfa exponent)
	int node1;
	int node2;
} HCluster;

class MaxMeaningfulClustering
{
public:
	
	/// Constructor.
	MaxMeaningfulClustering(unsigned char method, unsigned char metric);
	
	/// Does hierarchical clustering and detects the Max Meaningful Clusters
	/// @param[in] data The data feature vectors to be analyzed.
	/// @param[in] Num  Number of data samples.
	/// @param[in] dim  Dimension of the feature vectors.
	/// @param[in] method Clustering method.
	/// @param[in] metric Similarity metric for clustering.
	/// @param[out] meaningful_clusters Detected Max Meaningful Clusters.
	void operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector< vector<int> > *meaningful_clusters);
	void operator()(t_float *data, unsigned int num, unsigned char method, vector< vector<int> > *meaningful_clusters);

private:
	/// Helper function
	void build_merge_info(t_float *dendogram, t_float *data, int num, int dim, bool use_full_merge_rule, vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters);
	void build_merge_info(t_float *dendogram, int num, vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters);
	
	/// Number of False Alarms	
	int nfa(float sigma, float sigma2, int k, int N);

};

#endif
