#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:56:05 2019

@author: farig
"""
import os
import argparse

def create_dictionary(data_file_path, dict_to_save):
    print("creating vocabulary")
    
    lines = open(data_file_path).readlines()
    cui_set = set()
    user_list = []
    
    if not os.path.exists("../data/users/"): os.mkdir("../data/users/")        
    
    for line in lines:
        parts = line.strip().split(" ")
        user = parts[0].split("/")[1].split(".")[0]
        cuis = parts[1:]
        
        user_list.append(user)
        
        cui_dict = dict()
        
        for cui in cuis:
            if cui not in cui_dict:
                cui_dict[cui] = 1
            else:
                cui_dict[cui] += 1
                
            cui_set.add(cui)

	
        fw = open("../data/users/cuis_" + user + ".txt", "w")
        
        for cui in cui_dict:
            fw.write(cui + ": " + str(cui_dict[cui]) + "\n")
        
        fw.close()
        
    fw = open(dict_to_save, "w")    
        
    cuilist = list(cui_set)
    
    for i,cui in enumerate(cuilist):
        fw.write(cui + "\t" + str(i) + "\n")
    fw.close()
    
    fw = open("../user_list_full.txt", "w")
    
    for user_id in user_list:
        fw.write(user_id + "\n")
    fw.close()

def read_vocab(path_to_dict):
    print("reading vocabulary")
    lines = open(path_to_dict, "r").readlines()
    vocab = [0]*len(lines)
    for line in lines:
        parts = line.strip().split("\t")
        vocab[int(parts[1].strip())] = parts[0].strip()
        
    return vocab

def create_user_vectors(path_to_dict, path_to_data, path_to_save):
    fw = open(path_to_save, "w")
    
    print("creating user vectors")
    
    vocab = read_vocab(path_to_dict)
    
    fr = open("../user_list_full.txt", "r").readlines()
    
    userlist = []
    
    for line in fr:
        userlist.append(line.strip())
    
    print(len(userlist))
    
    user_vectors = {}
     
    for i,user in enumerate(userlist):
        print(i)
        vect = [0]*len(vocab)
        fr = open(path_to_data + "cuis_" + user + ".txt", "r").readlines()
        for line in fr:
            parts = line.strip().split(":")
            cui = parts[0].strip()
            count = int(parts[1].strip())
            vect[vocab.index(cui)] = count
   
        fw.write(str(vect[0]))
    
        for v in vect[1:]:
            fw.write("," + str(v))
    
        fw.write("\n")
        user_vectors[user] = vect
            
    fw.close()
    
def arg_parser():
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--cui_file",
                        default="../data/psychosis_cuis_processed_mclean.txt",
                        type=str,
                        required=True,
                        help="path to the CUI file (default: psychosis_cuis_processed_mclean.txt)"
                        )    
    args = parser.parse_args()
    return args

def main():    
    args = arg_parser()
    cui_file = args.cui_file
    
    if os.path.exists("../cuis_full.dict"):
        print("CUI dictionary exists")
        pass
    else:
        create_dictionary(cui_file, "../cuis_full.dict")
        
    if os.path.exists("../user_vectors_full"):
        print("Dense CUI user vector exists")
        pass
    else:
        create_user_vectors("../cuis_full.dict", "../data/users/", "../user_vectors_full")

    
    
if __name__ == "__main__":
    main()
