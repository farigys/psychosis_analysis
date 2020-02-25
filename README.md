### Files to create clustering on McLean Psychosis data.

#### Data files required (should be in the folder ../data):

1. CUIs extracted for each patient using CTAKES in one file (each line represents one patient)
2. Vector outputs from the DNN model (will be used for clustering)

#### Codes:

**1. preprocessing.py:** 

Creates dictionaries and vector files used for creating clusters.

**Command to run:** python preprocessor.py --cui_file <path_to_the_data_file_1>

**2. clustering.py:** 

Creates clusters using data file #2. Also creates files with TF-IDF for each CUI related to the clusters.

**Command to run:** python clustering.py --path <path_to_data_file_2> --clustering_technique <clust_tech>

##### Parameters:

###### Required:
1. --path: path to the file containing vector outputs from the DNN model
2. --clustering_technique: clustering technique you want to use- can be kmeans, dbscan or meanshift

###### Optional:
1. --viz_technique: Visualization choice- can be either pca or tsne
2. --cluster_count: required for Kmeans clustering. Default 2
3. --epsilon: Epsilon defined in the DBSCAN clustering algorithm. Default 10
4. --occurance_threshold: Minimum number of patient documents a CUI should occur to be considered in the TF-IDF analysis. Default 1
5. --save_clusters: yes to save the .png flies for the clustering
