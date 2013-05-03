
#include "max_meaningful_clustering.h"

MaxMeaningfulClustering::MaxMeaningfulClustering(unsigned char method, unsigned char metric)
{

}

void MaxMeaningfulClustering::operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector< vector<int> > *meaningful_clusters)
{
	
	t_float *Z = (t_float*)malloc(((num-1)*4) * sizeof(t_float)); // we need 4 floats foreach sample merge.
	linkage_vector(data, (int)num, dim, Z, method, metric);
	
	vector<HCluster> merge_info;
	build_merge_info(Z, data, (int)num, dim, false, &merge_info, meaningful_clusters);

	free(Z);
	merge_info.clear();
}

void MaxMeaningfulClustering::operator()(t_float *data, unsigned int num, unsigned char method, vector< vector<int> > *meaningful_clusters)
{
	
	t_float *Z = (t_float*)malloc(((num-1)*4) * sizeof(t_float)); // we need 4 floats foreach sample merge.
	linkage(data, (int)num, Z, method); //TODO think if complete linkage is the correct
	
	vector<HCluster> merge_info;
	build_merge_info(Z, (int)num, &merge_info, meaningful_clusters);

	free(Z);
	merge_info.clear();
}

void MaxMeaningfulClustering::build_merge_info(t_float *Z, t_float *X, int N, int dim, bool use_full_merge_rule, vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters)
{

	// walk the whole dendogram
	for (int i=0; i<(N-1)*4; i=i+4)
	{
		HCluster cluster;
		cluster.num_elem = Z[i+3]; //number of elements

		int node1  = Z[i];
		int node2  = Z[i+1];
		float dist = Z[i+2];
	
		if (node1<N)
		{
			vector<float> point;
			for (int n=0; n<dim; n++)
				point.push_back(X[node1*dim+n]);
			cluster.points.push_back(point);
			cluster.elements.push_back((int)node1);
		}
		else
		{
			for (int i=0; i<merge_info->at(node1-N).points.size(); i++)
			{
				cluster.points.push_back(merge_info->at(node1-N).points[i]);
				cluster.elements.push_back(merge_info->at(node1-N).elements[i]);
			}
			//update the extended volume of node1 using the dist where this cluster merge with another
			merge_info->at(node1-N).dist_ext = dist;
		}
		if (node2<N)
		{
			vector<float> point;
			for (int n=0; n<dim; n++)
				point.push_back(X[node2*dim+n]);
			cluster.points.push_back(point);
			cluster.elements.push_back((int)node2);
		}
		else
		{
			for (int i=0; i<merge_info->at(node2-N).points.size(); i++)
			{
				cluster.points.push_back(merge_info->at(node2-N).points[i]);
				cluster.elements.push_back(merge_info->at(node2-N).elements[i]);
			}

			//update the extended volume of node2 using the dist where this cluster merge with another
			merge_info->at(node2-N).dist_ext = dist;
		}

		Minibox mb;
		for (int i=0; i<cluster.points.size(); i++)
		{
			mb.check_in(&cluster.points.at(i));	
		}

		cluster.dist   = dist;
		cluster.volume = mb.volume();
		if (cluster.volume >= 1)
			cluster.volume = 0.999999;
		if (cluster.volume == 0)
			cluster.volume = 0.001; //TODO is this the minimum we can get?

		cluster.volume_ext=1;

		if (node1>=N)
		{
			merge_info->at(node1-N).volume_ext = cluster.volume;
		}
		if (node2>=N)
		{
			merge_info->at(node2-N).volume_ext = cluster.volume;
		}
	
		cluster.node1 = node1;	
		cluster.node2 = node2;	
	
		merge_info->push_back(cluster);

	}

	for (int i=0; i<merge_info->size(); i++)
	{

		merge_info->at(i).nfa = nfa(merge_info->at(i).volume, merge_info->at(i).volume_ext, merge_info->at(i).num_elem, N);
		int node1 = merge_info->at(i).node1;
		int node2 = merge_info->at(i).node2;

		{
			if ((node1<N)&&(node2<N))
			{
				//els dos nodes son single samples (nfa=1) per tant aquest cluster es maxim
				merge_info->at(i).max_meaningful = true;
				merge_info->at(i).max_in_branch.push_back(i);
				merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
			} else {
				if ((node1>=N)&&(node2>=N))
				{
					//els dos nodes son "sets" per tant hem d'avaluar el merging condition
					if ( ( (use_full_merge_rule) && ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + merge_info->at(node2-N).nfa) && (merge_info->at(i).nfa<min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch))) ) || ( (!use_full_merge_rule) && ((merge_info->at(i).nfa<min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch))) ) )
					{
						merge_info->at(i).max_meaningful = true;
                                		merge_info->at(i).max_in_branch.push_back(i);
                                		merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
						for (int k =0; k<merge_info->at(node1-N).max_in_branch.size(); k++)
							merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
						for (int k =0; k<merge_info->at(node2-N).max_in_branch.size(); k++)
							merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
					} else {
						merge_info->at(i).max_meaningful = false;
						merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node1-N).max_in_branch.begin(),merge_info->at(node1-N).max_in_branch.end());
						merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node2-N).max_in_branch.begin(),merge_info->at(node2-N).max_in_branch.end());
						if (merge_info->at(i).nfa<min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch))
							merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
						else 
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch);
					}
				} else {

					//un dels nodes es un "set" i l'altre es un single sample, s'avalua el merging condition pero amb compte
					if (node1>=N)
					{
						if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + 1) && (merge_info->at(i).nfa<merge_info->at(node1-N).min_nfa_in_branch))
						{
							merge_info->at(i).max_meaningful = true;
                                			merge_info->at(i).max_in_branch.push_back(i);
                                			merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
							for (int k =0; k<merge_info->at(node1-N).max_in_branch.size(); k++)
								merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
						} else {
							merge_info->at(i).max_meaningful = false;
							merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node1-N).max_in_branch.begin(),merge_info->at(node1-N).max_in_branch.end());
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,merge_info->at(node1-N).min_nfa_in_branch);
						}
					} else {
						if ((merge_info->at(i).nfa < merge_info->at(node2-N).nfa + 1) && (merge_info->at(i).nfa<merge_info->at(node2-N).min_nfa_in_branch))
						{
							merge_info->at(i).max_meaningful = true;
                                			merge_info->at(i).max_in_branch.push_back(i);
                                			merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
							for (int k =0; k<merge_info->at(node2-N).max_in_branch.size(); k++)
								merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
						} else {
							merge_info->at(i).max_meaningful = false;
							merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node2-N).max_in_branch.begin(),merge_info->at(node2-N).max_in_branch.end());
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,merge_info->at(node2-N).min_nfa_in_branch);
						}
					}
				}
			}


		} 

	}	

	for (int i=0; i<merge_info->size(); i++)
	{
		if (merge_info->at(i).max_meaningful)
		{
			vector<int> cluster;
			for (int k=0; k<merge_info->at(i).elements.size();k++)
				cluster.push_back(merge_info->at(i).elements.at(k));
			meaningful_clusters->push_back(cluster);
		}
	}	

}

void MaxMeaningfulClustering::build_merge_info(t_float *Z, int N, vector<HCluster> *merge_info, vector< vector<int> > *meaningful_clusters)
{

	// walk the whole dendogram
	for (int i=0; i<(N-1)*4; i=i+4)
	{
		HCluster cluster;
		cluster.num_elem = Z[i+3]; //number of elements

		int node1  = Z[i];
		int node2  = Z[i+1];
		float dist = Z[i+2];
		if (dist != dist) //this is to avoid NaN values
			dist=0;
	
		//fprintf(stderr," merging %d %d\n",node1,node2);

		if (node1<N)
		{
			cluster.elements.push_back((int)node1);
		}
		else
		{
			for (int i=0; i<merge_info->at(node1-N).elements.size(); i++)
			{
				cluster.elements.push_back(merge_info->at(node1-N).elements[i]);
			}
		}
		if (node2<N)
		{
			cluster.elements.push_back((int)node2);
		}
		else
		{
			for (int i=0; i<merge_info->at(node2-N).elements.size(); i++)
			{
				cluster.elements.push_back(merge_info->at(node2-N).elements[i]);
			}
		}

		cluster.dist   = dist;
		if (cluster.dist >= 1)
			cluster.dist = 0.999999;
		if (cluster.dist == 0)
			cluster.dist = 1.e-25; //TODO is this the minimum we can get?

		cluster.dist_ext   = 1;
		
		if (node1>=N)
		{
			merge_info->at(node1-N).dist_ext = cluster.dist;
		}
		if (node2>=N)
		{
			merge_info->at(node2-N).dist_ext = cluster.dist;
		}
	
		cluster.node1 = node1;	
		cluster.node2 = node2;	
		
	
		merge_info->push_back(cluster);

	}

	//print all merge info		
	//cout << "---------------------------------------------------------" << endl;
	//cout << "-- MERGE INFO ---- Evidence Accumulation " << endl;
	//cout << "---------------------------------------------------------" << endl;

	for (int i=0; i<merge_info->size(); i++)
	{

		merge_info->at(i).nfa = nfa(merge_info->at(i).dist, merge_info->at(i).dist_ext, merge_info->at(i).num_elem, N);
		int node1 = merge_info->at(i).node1;
		int node2 = merge_info->at(i).node2;

		{

			if ((node1<N)&&(node2<N))
			{
				//els dos nodes son single samples (nfa=1) per tant aquest cluster es maxim
				merge_info->at(i).max_meaningful = true;
				merge_info->at(i).max_in_branch.push_back(i);
				merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
				//fprintf(stderr,"%d = (%d,%d) els dos nodes son single samples (nfa=1) per tant aquest merge_info->at(i) es maxim min_nfa_in_branch = %d \n",i,node1-N,node2-N,merge_info->at(i).min_nfa_in_branch);
			} else {
				if ((node1>=N)&&(node2>=N))
				{
					//els dos nodes son "sets" per tant hem d'avaluar el merging condition
					if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + merge_info->at(node2-N).nfa) && (merge_info->at(i).nfa<min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch)))
					{
						//fprintf(stderr,"%d = (%d,%d) MAX because  merging condition 1 (%d < %d + %d ) && (%d<min(%d,%d))   \n",i,node1-N,node2-N,merge_info->at(i).nfa,merge_info->at(node1-N).nfa, merge_info->at(node2-N).nfa, merge_info->at(i).nfa, merge_info->at(node1-N).nfa,merge_info->at(node2-N).nfa);
						merge_info->at(i).max_meaningful = true;
                                		merge_info->at(i).max_in_branch.push_back(i);
                                		merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
						for (int k =0; k<merge_info->at(node1-N).max_in_branch.size(); k++)
							merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
						for (int k =0; k<merge_info->at(node2-N).max_in_branch.size(); k++)
							merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
						//fprintf(stderr," min_nfa_in_branch = %d \n",merge_info->at(i).min_nfa_in_branch);
					} else {
						merge_info->at(i).max_meaningful = false;
						merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node1-N).max_in_branch.begin(),merge_info->at(node1-N).max_in_branch.end());
						merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node2-N).max_in_branch.begin(),merge_info->at(node2-N).max_in_branch.end());
						if (merge_info->at(i).nfa<min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch))
							merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
						else 
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(node1-N).min_nfa_in_branch,merge_info->at(node2-N).min_nfa_in_branch);
						//fprintf(stderr,"%d = (%d,%d) NONmax  min_nfa_in_branch = %d \n",i,node1-N,node2-N,merge_info->at(i).min_nfa_in_branch);
					}
				} else {

					//un dels nodes es un "set" i l'altre es un single sample, s'avalua el merging condition pero amb compte
					if (node1>=N)
					{
						if ((merge_info->at(i).nfa < merge_info->at(node1-N).nfa + 1) && (merge_info->at(i).nfa<merge_info->at(node1-N).min_nfa_in_branch))
						{
						//fprintf(stderr,"%d = (%d,%d) MAX because  merging condition 2 (%d < %d + 1 ) && (%d<%d)   \n",i,node1-N,node2-N,merge_info->at(i).nfa,merge_info->at(node1-N).nfa, merge_info->at(i).nfa, merge_info->at(node1-N).min_nfa_in_branch);
							merge_info->at(i).max_meaningful = true;
                                			merge_info->at(i).max_in_branch.push_back(i);
                                			merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
							for (int k =0; k<merge_info->at(node1-N).max_in_branch.size(); k++)
								merge_info->at(merge_info->at(node1-N).max_in_branch.at(k)).max_meaningful = false;
						} else {
							merge_info->at(i).max_meaningful = false;
							merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node1-N).max_in_branch.begin(),merge_info->at(node1-N).max_in_branch.end());
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,merge_info->at(node1-N).min_nfa_in_branch);
							//fprintf(stderr,"%d = (%d,%d) NONmax2  min_nfa_in_branch = %d \n",i,node1-N,node2-N,merge_info->at(i).min_nfa_in_branch);
						}
					} else {
						if ((merge_info->at(i).nfa < merge_info->at(node2-N).nfa + 1) && (merge_info->at(i).nfa<merge_info->at(node2-N).min_nfa_in_branch))
						{
						//fprintf(stderr,"%d = (%d,%d) MAX because  merging condition 3 (%d < %d + 1 ) && (%d<%d)  \n ",i,node1-N,node2-N,merge_info->at(i).nfa,merge_info->at(node2-N).nfa, merge_info->at(i).nfa, merge_info->at(node2-N).min_nfa_in_branch);
							merge_info->at(i).max_meaningful = true;
                                			merge_info->at(i).max_in_branch.push_back(i);
                                			merge_info->at(i).min_nfa_in_branch = merge_info->at(i).nfa;
							for (int k =0; k<merge_info->at(node2-N).max_in_branch.size(); k++)
								merge_info->at(merge_info->at(node2-N).max_in_branch.at(k)).max_meaningful = false;
						} else {
							merge_info->at(i).max_meaningful = false;
							merge_info->at(i).max_in_branch.insert(merge_info->at(i).max_in_branch.end(),merge_info->at(node2-N).max_in_branch.begin(),merge_info->at(node2-N).max_in_branch.end());
							merge_info->at(i).min_nfa_in_branch = min(merge_info->at(i).nfa,merge_info->at(node2-N).min_nfa_in_branch);
							//fprintf(stderr,"%d = (%d,%d) NONmax3  min_nfa_in_branch = %d \n",i,node1-N,node2-N,merge_info->at(i).min_nfa_in_branch);
						}
					}
				}
			}


		} 
	}	

	for (int i=0; i<merge_info->size(); i++)
	{
		if (merge_info->at(i).max_meaningful)
		{
			vector<int> cluster;
			for (int k=0; k<merge_info->at(i).elements.size();k++)
				cluster.push_back(merge_info->at(i).elements.at(k));
			meaningful_clusters->push_back(cluster);
		}
	}	

}
	
int MaxMeaningfulClustering::nfa(float sigma, float sigma2, int k, int N)
{
    	return -1*(int)NFA( N, k, (double) sigma, 0); //this uses an approximation for the nfa calculations (faster)
}
