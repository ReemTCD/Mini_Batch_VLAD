"""
TODO: Evaluate VLAD and PBVLAD on Oxford5k dataset.
"""

import os
import pickle
from utils import read_files_recursive
from utils import load_pickle, save_pickle, load_h5, load_joblib, save_joblib, norm_feature_scale

import cv2
from sift import detect_sift
from mser import detect_mser

# using VLAD implementation from Jorge Guevara; jorjasso: https://github.com/jorjasso/VLAD
from VLADlib.VLAD import *
from VLADlib.Descriptors import *
from vlad import extract_sift_within_mser
from gen_data import generate_patch_data

from collections import defaultdict

from tqdm import tqdm

def get_image_names(gt_dir, file_list):
    image_names = []

    image_links = defaultdict(lambda: []) # dict for linking query images to retrieved ones

    for file_name in file_list:

        link_key = file_name.split('_')
        if len(link_key) == 4:
            link_key = "_".join(link_key[:2])
        else:
            link_key = link_key[0]

        with open(os.path.join(gt_dir, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if ' ' in line:
                    img_name = line.split(' ')[0] # only need the file name
                else:
                    img_name = line

                if '.jpg' not in img_name:
                    img_name += '.jpg'
                image_names.append(img_name)

                if 'oxc1_' in img_name:
                    img_name = img_name.replace('oxc1_','')

                image_links[link_key].append(img_name)

    for key in image_links.keys():
        image_links[key] = list(set(image_links[key]))

    image_names = sorted(list(set(image_names)))

    return image_names, image_links

# vlad variants
SELECTED_VLAD = 'orig-vlad' 
#SELECTED_VLAD = 'pb-vlad' 

# subset settings
SELECTED_SUBSET = 'all'
#SELECTED_SUBSET = 'good'
#SELECTED_SUBSET = '128im'
#SELECTED_SUBSET = '256ok'

# pca settings
#PERFORM_PCA = 1024
PERFORM_PCA = 2048 
#PERFORM_PCA = 4096
#PERFORM_PCA = 8192
#PERFORM_PCA = False

import numpy as np
np.random.seed(0)

def extractor(x):
    return {
        "orig-vlad": describeSIFT,
        "pb-vlad": extract_sift_within_mser
    }[x]


if __name__ == '__main__':
 
    data_dir = 'data/oxford5k/images/'
    gt_dir = 'data/oxford5k/gt' # ground truth txt files with lists of good, ok, junk, query image names

    #test_pickle = 'pickles/vlad_queries_formatted.pickle'
    #formatted = load_pickle(test_pickle, None)
    #print(formatted)
    #exit()

    #save_to = 'pickles/vlad_formatted_queries_{}.pickle'.format(SELECTED_SUBSET)
    #formatted = load_pickle(save_to, None)
    #print(formatted)
    #exit()

    img_list_pickle = './pickles/img_list_oxford.pickle'    
    #img_list = read_files_recursive(data_dir, fileNamesOnly=True, writeToFile=None)
    img_list = load_pickle(img_list_pickle, read_files_recursive, path=data_dir, fileNamesOnly=True, writeToFile=None)

    gt_list_pickle = './pickles/gt_list_oxford.pickle'    
    #img_list = read_files_recursive(data_dir, fileNamesOnly=True, writeToFile=None)
    gt_list = load_pickle(gt_list_pickle, read_files_recursive, path=gt_dir, fileNamesOnly=True, writeToFile=None)

    gt_class_dict = {}
    class_id = 0
    for item in gt_list:
        link_key = item.split('_')
        if len(link_key) == 4:
            link_key = "_".join(link_key[:2])
        else:
            link_key = link_key[0]

        if link_key not in gt_class_dict:
            gt_class_dict[link_key] = class_id
            class_id = class_id+1

    # separate lists for each oxford image group
    queries = [f for f in gt_list if '_query' in f]
    goodies = [f for f in gt_list if '_good' in f]
    okayies = [f for f in gt_list if '_ok' in f]
    junkies = [f for f in gt_list if '_junk' in f]

    queries, queries_links = get_image_names(gt_dir, queries)
    # why the hell they added this 'oxc1_' prefix to query image names!
    queries = [f if not 'oxc1_' in f else f.replace('oxc1_','') for f in queries]
    #queries = queries[0:2]

    goodies, goodies_links = get_image_names(gt_dir, goodies)
    okayies, okayies_links = get_image_names(gt_dir, okayies)
    junkies, junkies_links = get_image_names(gt_dir, junkies)
    
    import collections
    all_links = collections.defaultdict(list)
    for d in [goodies_links, okayies_links, junkies_links]:
        for k, v in d.items():  # d.items() in Python 3+
            for item in v:
                all_links[k].append(item)

    # the rest of the images do not have objects present
    baddies = [f for f in img_list if f not in queries and f not in goodies and f not in okayies and f not in junkies]

    print("Total query images: ", len(queries))
    print("First item: ", queries[0])
    print("Total good images: ", len(goodies))
    print("First item: ", goodies[0])
    print("Total okay images: ", len(okayies))
    print("First item: ", okayies[0])
    print("Total junk images: ", len(junkies))
    print("First item: ", junkies[0])
    print("Total bad images: ", len(baddies))
    print("First item: ", baddies[0])

    print("Total gt lists: ", len(gt_list))
    print("First gt item: ", gt_list[0])

    def f(x):
        return {
            "good":goodies,
            "okay":okayies,
            "junk":junkies,
            "demo":goodies[:10],
            "128im":goodies[:128],
            "256ok":okayies[:256],
            "all":goodies + okayies + junkies # + baddies
        }[x]


    def link(x):
        return {
            "good":goodies_links,
            "okay":okayies_links,
            "junk":junkies_links,
            "demo":goodies_links,
            "128im":goodies_links,
            "256ok":okayies_links,
            "all":all_links
        }[x]


    data = f(SELECTED_SUBSET)
    match_links = link(SELECTED_SUBSET)
    #print("Match links:", match_links)

    print("Precompute SIFT, MSER, {} on {} images. Proceed?".format(SELECTED_VLAD, SELECTED_SUBSET))

    print("Computing SIFT...")
    # 1. Compute SIFT descriptors from a dataset. The supported descriptors are ORB, SIFT and SURF:
    part = 'oxford5k_{}'.format(SELECTED_SUBSET)
    save_to = "pickles/{}_sifts_{}.h5".format(part, SELECTED_VLAD)
    patch_dict = {}
    if SELECTED_VLAD == 'pb-vlad':
        #print("images: ", data)
        sift_descriptors = load_h5(save_to, getPBDescriptors, data_dir, extractor(SELECTED_VLAD), optional_file_list=data)
        #sift_descriptors = load_h5(save_to, generate_patch_data, data_dir, data, patch_dict, rescale_images=False)    
        #save_to = "pickles/{}_patchdict_{}.h5".format(part, SELECTED_VLAD)
        #patch_dict = load_joblib(save_to, lambda data: data, patch_dict)
    else:
        sift_descriptors = load_h5(save_to, getDescriptors, data_dir, extractor(SELECTED_VLAD), optional_file_list=data)
    print("SIFT shape: ", sift_descriptors.shape)

    print("Constructing VISUAL dictionary...")
    # 2. Construct a visual dictionary from the descriptors in path -d, with -w visual words:
    number_of_visual_words = 128
    #visual_dictionary=kMeansDictionary(sift_descriptors, number_of_visual_words)
    #save_to = "pickles/vlad_dict_{}_{}.pickle".format(SELECTED_VLAD, part)
    #visual_dictionary = load_pickle(save_to, kMeansDictionary, sift_descriptors, number_of_visual_words)
    save_to = "pickles/{}_visual-dict_{}.joblib".format(part, 'orig-vlad') #, "orig-vlad") # test if orig-vlad dict helps: SELECTED_VLAD)
    visual_dictionary = load_joblib(save_to, kMeansDictionary, sift_descriptors, number_of_visual_words)
    #save_pickle(visual_dictionary, save_to)
    visual_dictionary.verbose = 0
    print(type(visual_dictionary))
    #input('Proceed?')

    print("Processing VLAD descriptors...")
    save_to = "pickles/{}_{}_pca={}_vlads.pickle".format(part, SELECTED_VLAD, PERFORM_PCA)
    
    patch2image = {}
    idImages = None
    patch2imageNP = None
    save_pca_to = "pickles/{}_{}_fittedPCA={}.pickle".format(part, SELECTED_VLAD, PERFORM_PCA)
    pca_label = PERFORM_PCA
    if type(PERFORM_PCA) == int:
        print("Fitting PCA on VLAD descriptors...")
        print(sift_descriptors.shape)
        n_components = PERFORM_PCA // sift_descriptors.shape[1] 
        PERFORM_PCA = PCA(n_components=n_components) # fit transform entire vlads
            
    
    if SELECTED_VLAD == 'pb-vlad':
        if type(PERFORM_PCA) is PCA: # False:
            # first only train pca
            if not os.path.exists(save_pca_to):
                _, _ = load_pickle(save_to+".tmp-pca-fit.pickle", getPBVLADDescriptors, data_dir, 
                    extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, 
                    perform_pca=PERFORM_PCA, train_only=True)
                save_pickle(save_pca_to, PERFORM_PCA)
            else:
                PERFORM_PCA = load_pickle(save_pca_to, None)
                # the use trained pca to transform Vlads
            print("Transforming VLAD descriptors with fitted PCA...")
            V, idImages = load_pickle(save_to, getPBVLADDescriptors, data_dir, 
                    extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, 
                    perform_pca=PERFORM_PCA, train_only=False)
                
        else:
            #'''
            V, idImages = load_pickle(save_to, getPBVLADDescriptors, data_dir, 
                extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, 
                perform_pca=PERFORM_PCA, train_only=False)
            '''
            # orig pbvlad imple
            V = load_joblib(save_to, getVLADDescriptors_per_image, 
                patch_dict, sift_descriptors, visual_dictionary, perform_pca=True)
            idImages = list(patch_dict.keys())
            #save_pickle([V, idImages, data_dir], save_to)
            '''
        '''        
        patch2imageNP = np.zeros((len(data), 3), dtype=int) # first is the min patch index, then max patch index, then image index
        idImages = data
        # map all patches to image ids
        prev_num_des = 0
        for ix, k in enumerate(data): #
            v = patch_dict[os.path.join(data_dir, k)]
            num_des = len(v[1]) + prev_num_des
            patch2image[(prev_num_des, num_des-1)] = k # os.path.basename(k)

            patch2imageNP[ix, 0] = prev_num_des
            patch2imageNP[ix, 1] = num_des-1
            patch2imageNP[ix, 2] = ix

            prev_num_des = num_des
        #print("Total items: ", patch2image.items())
        #print("Total images: ", len(idImages))
        #input("go_on?")
        '''
    else:
        if type(PERFORM_PCA) is PCA: # False:
            # PCA TEST
            # first only train pca
            if not os.path.exists(save_pca_to):
                _, _ = load_pickle(save_to+".pca-fit.pickle", getVLADDescriptors, data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, perform_pca=PERFORM_PCA, train_only=True)
                save_pickle(save_pca_to, PERFORM_PCA)
            else:
                PERFORM_PCA = load_pickle(save_pca_to, None)
                
            print("Transforming VLAD descriptors with fitted PCA...")
            V, idImages = load_pickle(save_to, getVLADDescriptors, data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, perform_pca=PERFORM_PCA, train_only=False)
        else:
            V, idImages = load_pickle(save_to, getVLADDescriptors, data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=data, perform_pca=PERFORM_PCA, train_only=False)
            
    print("V shape: ", V.shape)
    #print("Id imageS:", idImages)
    #input('Proceed?')
    

    print("Making TREE index...")
    # 4. Make an index from VLAD descriptors using a ball-tree DS:
    leafSize = 20 # TODO: check this
    #tree = indexBallTree(V,leafSize)
    save_to = "pickles/{}_indextree_{}.pickle".format(part, SELECTED_VLAD)
    tree = load_pickle(save_to, indexBallTree, V, leafSize) # V
    #save_pickle(tree, save_to)
    print("Tree:", tree)

    # 5. Evaluation Query:
    descriptorName = 'SIFT'
    data_dir = 'data/oxford5k/images/'
    query_results = {}

    for img_name in tqdm(queries):
        img_path = os.path.join(data_dir, img_name)
        
        # TODO: calculate VLADS and sifts
        #sift_descriptors = load_h5(save_to, generate_patch_data, data_dir, data, patch_dict, rescale_images=False)       
        
        print(img_path)
        if SELECTED_VLAD == 'orig-vlad':
            # querying by matching the lists
            query_V, _ = getVLADDescriptors(data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=[img_name], perform_pca=PERFORM_PCA, train_only=False)
            dist, ind = find_and_sort(query_V, V, k=20)
            # querying by searching the tree
            #dist,ind = queryWithFun(img_path, 10, extractor(SELECTED_VLAD), visual_dictionary, tree, perform_pca=False)
        elif SELECTED_VLAD == 'pb-vlad':
            query_patch_dict = {}
            # querying by matching the lists (per patch)
            '''
            descriptors = generate_patch_data(data_dir, [img_name], query_patch_dict, rescale_images=False)
            #print("sift shape: ", descriptors.shape)
            query_V = VLAD(descriptors, visual_dictionary, perform_pca=PERFORM_PCA)
            #query_V, _ = getVLADDescriptors(data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=[img_name], perform_pca=pca, train_only=False)
            '''
            # querying by matching the lists (per image)
            query_V, query_imageIDs = getPBVLADDescriptors(data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=[img_name], perform_pca=PERFORM_PCA, train_only=False)
            #'''
            _, query_indices = np.unique(np.array(query_imageIDs), return_inverse=True)
            _, ref_indices = np.unique(np.array(idImages), return_inverse=True)
            dist, ind = find_and_sort(query_V, V, k=20, query_ids=query_indices, ref_ids=ref_indices) #, patch2imageNP)

            # querying by searching the tree
            #dist,ind = queryWithFun(img_path, 10, extractor(SELECTED_VLAD), visual_dictionary, tree, perform_pca=False)
        
        query_results[img_name] = (list(dist), list(ind))
    
    save_to = 'pickles/oxford5k_{}_pca={}_{}_queries.pickle'.format(SELECTED_SUBSET, pca_label, SELECTED_VLAD)
    save_pickle(save_to, query_results)
  
    query_class_results = {}
    for img_name in tqdm(queries):
        path = os.path.join(data_dir, img_name)
        dist, ind = query_results[img_name]
        
        # TODO: query gt lists and find matches
        query_key = None
        match_key = None

        #print("elen keys: ", len(list(queries_links.keys())))
        #input('?')
        for key in queries_links.keys():
            name = os.path.basename(img_name)
            #print(queries_links[key], key, name)
            if name in queries_links[key]:
                query_key = key
                #input('?')
                break
                

        gt_matches = [None]*len(ind) 
        
        for key in match_links.keys():
            for i in ind:
                orig_i = i
                #for i in range(len(ind)): # DEBUG: this reports mAP: 0.12 using orig VLAD
                name = os.path.basename(idImages[i]) #.split('.')[0].split('_0')[0]
                #print(match_links[key], idImages[i], i, name)
                
                if name in match_links[key]:
                    match_key = key
                    gt_matches[ind.index(orig_i)] = match_key

        query_class_id = gt_class_dict[query_key]
        result_class_ids = [gt_class_dict[match_key] for match_key in gt_matches]
        query_class_results[img_name] = (dist, result_class_ids, query_class_id)
        print("dists & predictions: ", query_class_results[img_name])
        
    save_to = 'pickles/oxford5k_{}_pca={}_{}_formatted_queries.pickle'.format(SELECTED_SUBSET, pca_label, SELECTED_VLAD)
    save_pickle(save_to, query_class_results)
