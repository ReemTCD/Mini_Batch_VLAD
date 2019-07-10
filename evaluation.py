#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This scripts runs the evaluation based on the stored pickles:
    1. Runs prepareData.py: this will read calculated pickles and prepare Y_test and Y_score needed to run mAP.py. It needs number of classes!
    2. Runs mAP.py which calculates precision-recall curve and calculates mAP based on Y_test and Y_score.
"""

# Import of the two files withing the evaluation directory
from evaluation import PrepareData
from evaluation import MeanAP
import math
import os
import numpy as np
from utils import load_joblib, load_pickle, read_files_recursive, distance_m
from collections import defaultdict

from config import _RESULTS_LABEL


def evaluate():

    eval_all = True
    eval_oxford = False
    
    if eval_all:
        ref_data_dir = 'data/images/' # directory for reference dataset (retrieving image set)
        ref_img_list_pickle = './pickles/pitt_list_images.pickle'    
        ref_img_list = load_pickle(ref_img_list_pickle, read_files_recursive, path=ref_data_dir, fileNamesOnly=False, writeToFile=None)

        print('>> OVERALL EVALUATION')
        _DIST_THR = 250 
        
        
        SELECTED_VLAD = 'Mini-batch-vlad'
        SELECTED_FEATURES = 'all'
        part = _RESULTS_LABEL#'par-partes'
        save_to = 'pickles/{}_{}_{}_{}_query-results.pickle'.format(_RESULTS_LABEL, SELECTED_VLAD, SELECTED_FEATURES, 'single')
        query_results_all = load_joblib(save_to, None)
        
        perc_all = PrepareData.overall_evaluation(query_results_all, ref_img_list, SELECTED_VLAD, SELECTED_FEATURES, _DIST_THR)
        MeanAP.plotPercentages(perc_all, SELECTED_FEATURES, show=False, title="{} - search area:{}m".format(SELECTED_VLAD, _DIST_THR))
        
        _DIST_THR = 125
        perc_all = PrepareData.overall_evaluation(query_results_all, ref_img_list, SELECTED_VLAD, SELECTED_FEATURES, _DIST_THR)
        MeanAP.plotPercentages(perc_all, SELECTED_FEATURES, show=False, title="{} - search area:{}m".format(SELECTED_VLAD, _DIST_THR))
        
        _DIST_THR = 250
        perc_all = PrepareData.overall_evaluation(query_results_all, ref_img_list, SELECTED_VLAD, SELECTED_FEATURES, _DIST_THR)
        MeanAP.plotPercentages(perc_all, SELECTED_FEATURES, show=False, title="{} - search area:{}m".format(SELECTED_VLAD, _DIST_THR))
        
        exit()
        SELECTED_FEATURES = 'select'
        save_to = 'pickles/{}_{}_{}_{}_query-results.pickle'.format(_RESULTS_LABEL, SELECTED_VLAD, SELECTED_FEATURES, 'single')
        query_results_select = load_joblib(save_to, None)
        SELECTED_FEATURES = 'random'
        perc_select = PrepareData.overall_evaluation(query_results_select, ref_img_list, SELECTED_VLAD, SELECTED_FEATURES)
        MeanAP.plotPercentages(perc_select, SELECTED_FEATURES, show=True, title=part + '-' + SELECTED_FEATURES)
        exit()

    if eval_oxford:
        print('>> Mini-batch-VLAD EVALUATION')
        SELECTED_VLAD = 'Mini-batch-vlad'
        SELECTED_SUBSET = 'all'
        
        # used to generate all numericc results in the report
        #pca_label = 1024
        pca_label = 2048
        #pca_label = 4096
        #pca_label = 8192
        #pca_label = False
        save_to = 'pickles/oxford5k_{}_pca={}_{}_formatted_queries.pickle'.format(SELECTED_SUBSET, pca_label, SELECTED_VLAD)
        print('>> ACCURACY CALCULATION')
        acc = PrepareData.calcAcc(save_to, 11)
        print(acc)
        Y_test, Y_score, n_classes = PrepareData.loadAndFormat(save_to, 11)
        print('>> Mini-batch-VLAD mAP CALCULATION')
        MeanAP.plotPRCMulti(Y_test, Y_score, n_classes, '{}-{}'.format(SELECTED_VLAD, "no-PCA" if not pca_label else pca_label))
        
        print('>> Mini-batch-VLAD EVALUATION')
        SELECTED_VLAD = 'Mini-batch-VLAD'
        save_to = 'pickles/oxford5k_{}_pca={}_{}_formatted_queries.pickle'.format(SELECTED_SUBSET, pca_label, SELECTED_VLAD)
        print('>> ACCURACY CALCULATION')
        acc = PrepareData.calcAcc(save_to, 11)
        print(acc)
        Y_test, Y_score, n_classes = PrepareData.loadAndFormat(save_to, 11)
        #print(Y_test.shape)
        #print(Y_score.shape)
        #exit()
        print('>> Orig mAP CALCULATION')
        MeanAP.plotPRCMulti(Y_test, Y_score, n_classes, '{}-{}'.format(SELECTED_VLAD, "no-PCA" if not pca_label else pca_label ))

def main(*args):
    evaluate()
    return 0

# Call main if used as a script
if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
