#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:37:37 2019

@author: farig
"""

from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns
from sklearn.decomposition import PCA as sklearnPCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import os
import argparse
import pickle
import math
from collections import Counter

def visualize(X_scaled, y, clust_tech, params):
    transformed = None
    viz = params["viz"]
    save = params["save"]
    if clust_tech == "dbscan": p = params["eps"]
    if clust_tech == "kmeans": p = params["k"]
    
    
    if viz == "pca":
        pca = sklearnPCA(n_components=2) 
        transformed = pd.DataFrame(pca.fit_transform(X_scaled))
            
    elif viz == "tsne":
        tsne = TSNE(n_components=2)
        transformed = pd.DataFrame(tsne.fit_transform(X_scaled))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    labels = np.unique(y)    
    
    transformed.insert(2, "y", y)
    
    for i,n_ in enumerate(labels):
        ax.scatter(transformed[y==n_][0], transformed[y==n_][1], label='Class '+str(n_))
            
    ax.legend()
            
    if save == "yes":
        if not os.path.exists("../clusters"): os.mkdir("../clusters")
        if not os.path.exists("../clusters/" + clust_tech + "_" + viz): os.mkdir("../clusters/" + clust_tech + "_" + viz)
        if clust_tech == "meanshift": path_to_write = "../clusters/meanshift_" + viz + "/meanshift_clusters.png" 
        else: path_to_write = "../clusters/" + clust_tech + "_" + viz + "/" + clust_tech + "_full_clusters_" + viz + "_" + str(p) + ".png"     
        fig.savefig(path_to_write)
        
    plt.show()    
    plt.close()
        

def dbscan_clustering(X_scaled, params):
    eps = params["eps"]
    clustering = DBSCAN(eps=eps, min_samples=2).fit(X_scaled)
    y = clustering.labels_
    
    if params["viz"]:
        visualize(X_scaled, y, "dbscan", params)
    
    return y
    

def kmeans_clustering(X_scaled, params):
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', 
                    max_iter=500, n_clusters=params["k"], n_init=10, n_jobs=1, 
                    precompute_distances='auto', random_state=42, tol=0.0001, verbose=0)

    y = kmeans.fit_predict(X_scaled)
        
    if params["viz"]:
        visualize(X_scaled, y, "kmeans", params)
    
    return y
    
def meanshift_clustering(X_scaled, params):
    y = MeanShift(min_bin_freq=3).fit_predict(X_scaled)
    
    if params["viz"]:
        visualize(X_scaled, y, "meanshift", params)
    
    return y
    
def read_cui_dict(path_to_dict):
    lines = open(path_to_dict, "r").readlines()
    cui_dict = []
    for line in lines:
        parts = line.strip().split("\t")
        cui_dict.append(parts[0].strip())
        
    return cui_dict

def create_and_visualize_clusters(X, clustering_technique, clust_params):
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = None
    
    if clustering_technique == "dbscan":
        y = dbscan_clustering(X_scaled, clust_params)
    
    if clustering_technique == "kmeans":
        y = kmeans_clustering(X_scaled, clust_params)
    
    if clustering_technique == "meanshift":
        y = meanshift_clustering(X_scaled, clust_params)
    
    counter = Counter(y)
    print(counter)

    return y

def cluster_analysis(users, y, clustering):
    user_cluster = {} 
    for z in zip(users,y):
        user_cluster[z[0]] = z[1]
    
    cuis = read_cui_dict("../cuis_full.dict")
    
    
    df = pd.read_csv("../user_vectors_full", header=None)
    df.columns = cuis
    df["user_id"] = users
    df["cluster"] = y
    
    for y_ in np.unique(y):
        print(y_)
        df_ = df[df["cluster"] == y_]
        #print(df_.shape)
        df_ = df_.drop(["cluster","user_id"], axis=1).T
        df_["Total"] = df_.sum(axis=1)
        totals = df_["Total"].tolist()
        print(len(totals))
        df_ = df_.drop(["Total"], axis=1)
        df_ = df_.T
        df_ = df_.describe().T
        df_["Total"] = totals
        df_ = df_.sort_values(by="Total", ascending=False)
        print(df_)
        if not os.path.exists("../cui_analysis"): os.mkdir("../cui_analysis")
        df_.to_csv("../cui_analysis/" + clustering + "_cluster_" + str(y_) + ".csv")
        #df_ = df_.T.sort_values("count")
        #df_ = df_["count", "mean", "std", "min", "max"]

    
def cluster_tfidf(users, y, clustering, threshold, cui_names):
    labels = np.unique(y)
    
    cuis = read_cui_dict("../cuis_full.dict")
    
    file = open("../user_vectors_full", "r")
    lines = file.readlines()
    
    tf = dict()
    idf = dict()
    tf_idf = dict()
    patient_count = dict()
    
    for cui in cuis:
        tf[cui] = [0]*len(labels)
        idf[cui] = set()
        tf_idf[cui] = [0.0]*len(labels)
        patient_count[cui] = 0
    
    for linec, line in enumerate(lines):
        clust = y[linec]
        
        parts = line.strip().split(",")
        
        #print(len(parts))
        
        for cp, part in enumerate(parts):
            if part == "0":
                continue
            else:
                cui = cuis[cp]
                tf[cui][clust] += int(part)
                idf[cui].add(clust)
                patient_count[cui]+=1
    
    #print(tf)
    
    #print(len(tf), len(idf), len(tf_idf))
            
    D = len(labels)
    
    if not os.path.exists("../clusters"): os.mkdir("../clusters")
    writefile = open("../clusters/tf_idf_clusters_" + clustering + "_" + str(D) + "_full.csv", "w")
    
    for cui in tf:
        if patient_count[cui] < threshold: continue

        if cui in cui_names: name = cui_names[cui]
        else: name = ""

        writefile.write(cui + "\t" + name)
        for i in range(D):
            tf_idf[cui][i] = tf[cui][i] * math.log((D*1.0)/len(idf[cui]))
            writefile.write("\t" + str(tf_idf[cui][i]))
        writefile.write("\t" + str(patient_count[cui]) + "\n")
        
    writefile.close()
    
    
    tfidffile = open("../clusters/tf_idf_file_" + str(D) + "_full.pkl", "wb")
    
    pickle.dump(tf_idf, tfidffile)
    
    tfidffile.close()
        
def read_cui_names(cui_file):
    lines = open(cui_file, "r").readlines()
    cui_name_dict = {}

    for line in lines:
        parts = line.strip().split("|")
        cui = parts[0]
        name = parts[14]
        
        if cui in cui_name_dict:
            continue
        cui_name_dict[cui] = name
        
    return cui_name_dict

def get_filtered_cui_list(filter_file):
    cui_set = set()
    
    lines = open(filter_file, "r").readlines()
    
    for line in lines:
        cui = line.strip()
        cui_set.add(cui)
        cui_set.add("-" + cui)
        
    return list(cui_set)
    

def arg_parser():
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--path",
                        default=None,
                        type=str,
                        required=True,
                        help="path to the vector file (vector elements should be separated by space)"
                        )
     
    parser.add_argument("--clustering_technique",
                        default=None,
                        type= str,
                        required=True,
                        help="select one: kmeans, dbscan, meanshift"
                        )
    
    parser.add_argument("--viz_technique",
                        default=None,
                        type= str,
                        required=False,
                        help="select one: pca, tsne"
                        )
    
    parser.add_argument("--cluster_count",
                        default=2,
                        type= int,
                        required=False,
                        help="provide the number of clusters for kmeans (default: 2)"
                        )
    
    parser.add_argument("--epsilon",
                        default=10,
                        type= float,
                        required=False,
                        help="provide the epsilon for dbscan clustering (default: 10)"
                        )
    
    parser.add_argument("--occurance_threshold",
                        default=1,
                        type= int,
                        required=False,
                        help="provide the minimum numbers of patients a CUI has to occur for to be counted"
                        )
    
    parser.add_argument("--save_clusters",
                        default=None,
                        type= str,
                        required=False,
                        help="yes to save clusters in file"
                        )
     
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    params = {}
    vector_file_path = args.path.lower()
    clustering = args.clustering_technique.lower()
    print(clustering)
    params["viz"] = None
    if args.viz_technique: params["viz"] = args.viz_technique.lower()
    params["k"] = None
    if args.cluster_count: params["k"] = args.cluster_count
    params["eps"] = None
    if args.epsilon: params["eps"] = args.epsilon
    params["save"] = None
    if args.save_clusters: params["save"] = args.save_clusters.lower()
    
    threshold = args.occurance_threshold
    
    #print(vector_file_path)
    
    #print(pd)
    
    df = pd.read_csv(vector_file_path, sep = " ", header = None)
    
    #print(df.shape())
    
    users_ = df[df.columns[0]].tolist()
    
    users = [u.split(":")[0] for u in users_]
    
    #print(users)
    
    df = df.drop(df.columns[0], axis=1)
    
    X = np.array(df.astype(float))
    
    #print(X.shape)
    
    print(params)
    
    y = create_and_visualize_clusters(X, clustering, params)
    
    #cluster_analysis(users, y, clustering)
    
    cui_names = read_cui_names("../MRCONSO.RRF")
    
    cluster_tfidf(users, y, clustering, threshold, cui_names)
    
    
if __name__ == "__main__":
    main()
