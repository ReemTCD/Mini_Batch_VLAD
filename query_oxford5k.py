"""
TODO: Evaluate VLAD and PBVLAD on Oxford5k dataset.
"""

import os
from os.path import exists, join
from utils import read_files_recursive, load_pickle, load_joblib
import pickle

import cv2
from sift import detect_sift
from mser import detect_mser

# using implementation from  Jorge Guevara; jorjasso: https://github.com/jorjasso/VLAD
from VLADlib.VLAD import *
from VLADlib.Descriptors import *
from vlad import extract_sift_within_mser

from eval_oxford5k import get_image_names, save_pickle, SELECTED_SUBSET, SELECTED_VLAD, PERFORM_PCA, extractor

if __name__ == '__main__':
 
    demo_path = 'demo/oxford5k-retrieved/'

    data_dir = 'data/oxford5k/images/'
    gt_dir = 'data/oxford5k/gt' # ground truth txt files with lists of good, ok, junk, query image names

    img_list_pickle = './pickles/img_list_oxford.pickle'    
    #img_list = read_files_recursive(data_dir, fileNamesOnly=True, writeToFile=None)
    img_list = load_pickle(img_list_pickle, read_files_recursive, path=data_dir, fileNamesOnly=True, writeToFile=None)

    gt_list_pickle = './pickles/gt_list_oxford.pickle'    
    #img_list = read_files_recursive(data_dir, fileNamesOnly=True, writeToFile=None)
    gt_list = load_pickle(gt_list_pickle, read_files_recursive, path=gt_dir, fileNamesOnly=True, writeToFile=None)

    # separate lists for each oxford image group
    queries = [f for f in gt_list if '_query' in f]
    goodies = [f for f in gt_list if '_good' in f]
    okayies = [f for f in gt_list if '_ok' in f]
    junkies = [f for f in gt_list if '_junk' in f]

    queries, queries_links = get_image_names(gt_dir, queries)
    # why they added this 'oxc1_' prefix to query image names!
    queries = [f if not 'oxc1_' in f else f.replace('oxc1_','') for f in queries]

    goodies, goodies_links = get_image_names(gt_dir, goodies)
    okayies, okayies_links = get_image_names(gt_dir, okayies)
    junkies, junkies_links = get_image_names(gt_dir, junkies)

    print("Total query images: ", len(queries))
    print("First item: ", queries[0])
    print("Total good images: ", len(goodies))
    print("First item: ", goodies[0])
    print("Total okay images: ", len(okayies))
    print("First item: ", okayies[0])
    print("Total junk images: ", len(junkies))
    print("First item: ", junkies[0])
    print("Total gt lists: ", len(gt_list))
    print("First gt item: ", gt_list[0])

    part = 'oxford5k_{}'.format(SELECTED_SUBSET)
    # load visual dict
    save_to = "pickles/{}_visual-dict_{}.joblib".format(part, 'orig-vlad')
    visual_dictionary = load_joblib(save_to, None)
    
    # load existing VLADs into V:
    save_to = "pickles/{}_{}_pca={}_vlads.pickle".format(part, SELECTED_VLAD, PERFORM_PCA)
    V, idImages = load_pickle(save_to, None)
    
    # load PCA
    from sklearn.decomposition import PCA
    save_pca_to = "pickles/{}_{}_fittedPCA={}.pickle".format(part, SELECTED_VLAD, PERFORM_PCA)
    PERFORM_PCA = load_pickle(save_pca_to, None)

    for img_name in tqdm(queries):
        img_path = os.path.join(data_dir, img_name)

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
            query_V, _ = getVLADDescriptors(data_dir, extractor(SELECTED_VLAD), visual_dictionary, optional_file_list=[img_name], perform_pca=PERFORM_PCA, train_only=False)
            #'''
            dist, ind = find_and_sort(query_V, V, k=20) #, patch2imageNP)
            
            # querying by searching the tree
            #dist,ind = queryWithFun(img_path, 10, extractor(SELECTED_VLAD), visual_dictionary, tree, perform_pca=False)
            
        # prepare directories:
        if demo_path is not None:
            #query_name = os.path.split(query_path)[-1]
            query_save = img_name[:-4]
            if not exists(join(demo_path, query_save)):
                #os.makedirs(join(demo_path, query_save, 'positive'))
                os.makedirs(join(demo_path, query_save, 'retrieved'))
                #os.makedirs(join(demo_path, query_save, 'actual'))
                
            img = cv2.imread(img_path)
            cv2.imwrite(join(demo_path, query_save, img_name), img)
        

        # loop over the results
        for rank, i in enumerate(ind):                 
            if demo_path is not None:
                img = cv2.imread(idImages[i])
                #img = cv2.resize(img, (640, 480))
                #print("QUERY RES: ", img_path, " > ", img.shape)
                save_path = join(demo_path, query_save,'retrieved', '{:02d}_'.format(rank+1)+os.path.split(idImages[i])[1])
                print("SAVED: ", save_path)
                cv2.imwrite(save_path, img)
            else:
                # load the result image and display it
                print("Image ID: ", idImages[i])
                result = cv2.imread(idImages[i])
                cv2.imshow("Result", result)
                cv2.waitKey(0)