# implementation of VLAD for CBIR(# Jorge Guevara, jorged@br.ibm.com)


import numpy as np 
import itertools
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2
from VLADlib.Descriptors import *
import os
from tqdm import tqdm

#from memory_profiler import profile

# inputs
# getDescriptors for whole dataset

# functionHandleDescriptor={describeSURF, describeSIFT, describeORB}


#import hdbscan
# input
# training = a set of descriptors
def  kMeansDictionary(training, k, pretrained_object = None):

    #K-means algorithm -- original implementation
    #est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1, n_jobs=-1).fit(training)

    MiniBatch K-means algorithm (much faster)
    if len(training.shape) == 1:
       training = training.reshape(1, -1)  # we need to reshape like if training contains single sample

    if pretrained_object is not None:
        est = pretrained_object.fit(training)
    else:
        #pretrained_object = KMeans(n_clusters=k,init='k-means++',max_iter=300,tol=0.00001, n_jobs=-1, verbose=1)
        pretrained_object = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, batch_size=500000, verbose=1, max_no_improvement=None, reassignment_ratio=0.99) 
        #pretrained_object = hdbscan.HDBSCAN()
        #pretrained_object = AgglomerativeClustering(n_clusters=k)
        # pretrained_object.fit(blobs)
        #
        est = pretrained_object.fit(training)

    

    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est
    #clf2 = pickle.loads(s)

# compute vlad descriptors for te whole dataset
# input: path = path of the dataset
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm


def getVLADDescriptors_per_image(patch_dict, sift_descriptors, visualDictionary, img_list=None, perform_pca=True):
    
    max_des = 0
    filtered_names = []

    #print("Patch dict keys: ", len(patch_dict.keys()))

    t = lambda x: tqdm(x)
    if img_list is not None:
        if len(img_list) == 1:
            t = lambda x: x

    for img_name in t(img_list) if img_list is not None else t(patch_dict.keys()):
        if img_name in patch_dict.keys():
            max_des += 1 #len(patch_dict[img_name][1])

    #print("Max DES:" , max_des)
    #input("Cont?")

    if perform_pca is False:
        descriptors = np.zeros((max_des, 16384), dtype=float)
    else:
        if type(perform_pca) == PCA:
            n = perform_pca.n_components_
            descriptors = np.zeros((max_des, 128*n), dtype=float)
        else:
            descriptors = np.zeros((max_des, 2048), dtype=float)

    i = 0
    j = 0
    for img_name in t(img_list) if img_list is not None else t(patch_dict.keys()):

        if img_name not in patch_dict.keys():
            continue

        all_des = 0
        for num_des in patch_dict[img_name][1]:
            all_des += num_des
        
        descriptors[j,:] = VLAD(sift_descriptors[i:i+all_des,:],visualDictionary, perform_pca=perform_pca)
        
        i+=all_des
        j+=1

    return descriptors#, img_list if img_list is not None else patch_dict.keys()
    

#@profile
def getVLADDescriptors_v2(patch_dict, sift_descriptors, visualDictionary, img_list=None, perform_pca=True):
    
    max_des = 0
    filtered_names = []

    #print("Patch dict keys inside VLAD; ", len(patch_dict.keys()))

    #print("Estimating amount of VLADs... ")
    
    t = lambda x: tqdm(x)
    if img_list is not None:
        if len(img_list) == 1:
            t = lambda x: x

    for img_name in t(img_list) if img_list is not None else t(patch_dict.keys()):
        #print(img_name)
        if img_name in patch_dict.keys():
            max_des += len(patch_dict[img_name][1])
    #print("Total amount of VLADs: ", max_des)

    if not perform_pca:
        descriptors = np.zeros((max_des, 16384), dtype=float)
    else:
        if type(perform_pca) == PCA:
            n = perform_pca.n_components_
            descriptors = np.zeros((max_des, 128*n), dtype=float)
        else:
            descriptors = np.zeros((max_des, 2048), dtype=float)
    
    i = 0
    j = 0
    for img_name in t(img_list) if img_list is not None else t(patch_dict.keys()):
        #print(type(des))
        if img_name not in patch_dict.keys():
            continue
        for num_des in patch_dict[img_name][1]:
            descriptors[j,:] = VLAD(sift_descriptors[i:i+num_des,:],
                                    visualDictionary, perform_pca=perform_pca)
            i+=num_des
            j+=1
    
    return descriptors #, idImage
    


    #list to array    
    #descriptors = np.asarray(descriptors)
    return all_descriptors, idImage


def getVLADDescriptors(path,functionHandleDescriptor,visualDictionary, optional_file_list=None, perform_pca=False, train_only=False, resize_images=True):
    descriptors=list()
    all_descriptors = None # np.zeros((0, 128), dtype=float) # for SIFT:
    
    idImage =list()
    for imagePath in tqdm(optional_file_list) if optional_file_list is not None else tqdm(glob.glob(path+"/*.jpg")):
        #print(imagePath)
        if optional_file_list is not None and path not in imagePath:
            imagePath = os.path.join(path, imagePath)

        im=cv2.imread(imagePath)
        if resize_images:
            im=cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
        
        kp,des = functionHandleDescriptor(im)
        #if des!=None:
        if type(des) is np.ndarray and des.shape[0] > 0:
            v=VLAD(des,visualDictionary, perform_pca=perform_pca, train_only=train_only)
            
            if all_descriptors is None:
                v = np.expand_dims(v, axis=0)
                all_descriptors = v.copy().astype(np.float32)
                #print(all_descriptors.shape)
                #input("raW?")
                del v
            else:
                j, d = all_descriptors.shape
                max_desc = all_descriptors.shape[0] + 1 #v.shape[0]
                all_descriptors.resize((max_desc, d), refcheck=True) # works)
                all_descriptors[j:j+v.shape[0], :] = v.copy().astype(np.float32)
                del max_desc
                del v
            
            #descriptors.append(v)
            if type(kp) is int:
                idImage += [imagePath]*kp
            else:
                idImage.append(imagePath)

    #list to array    
    #descriptors = np.asarray(descriptors)
    return all_descriptors, idImage

# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

# compute vlad descriptors per PDF for te whole dataset, f
# input: path = dataset path
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm

def getVLADDescriptorsPerPDF(path,functionHandleDescriptor,visualDictionary):
    descriptors=list()
    idPDF =list()
    desPDF= list()

    #####
    #sorting the data
    data=list()
    for e in glob.glob(path+"/*.jpg"):
        #print("e: {}".format(e))
        s=e.split('/')
        #print("s: {}".format(s))
        s=s[1].split('-')
        #print("s: {}".format(s))
        s=s[0].split('.')
        #print("s: {}".format(s))
        s=int(s[0]+s[1])
        #print("s: {}".format(s))

        data.append([s,e])

    data=sorted(data, key=lambda atr: atr[0])
    #####

    #sFirst=glob.glob(path+"/*.jpg")[0].split('-')[0]
    sFirst=data[0][0]
    docCont=0
    docProcessed=0
    #for imagePath in glob.glob(path+"/*.jpg"):
    for s, imagePath in data:
        #print(imagePath)
        #s=imagePath.split('-')[0]
        #print("s : {}".format(s))
        #print("sFirst : {}".format(sFirst))

        #accumulate all pdf's image descriptors in a list
        if (s==sFirst):
            
            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)   
            
        else:
            docCont=docCont+1
            #compute VLAD for all the descriptors whithin a PDF
            #------------------
            if len(desPDF)!=0: 
                docProcessed=docProcessed+1
                #print("len desPDF: {}".format(len(desPDF)))
                #flatten list       
                desPDF = list(itertools.chain.from_iterable(desPDF))
                #list to array
                desPDF = np.asarray(desPDF)
                #VLAD per PDF
                v=VLAD(desPDF,visualDictionary)     
                descriptors.append(v)
                idPDF.append(sFirst)
            #------------------
            #update vars
            desPDF= list()
            sFirst=s
            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)

    #Last element
    docCont=docCont+1
    if len(desPDF)!=0: 
        docProcessed=docProcessed+1
        desPDF = list(itertools.chain.from_iterable(desPDF))
        desPDF = np.asarray(desPDF)
        v=VLAD(desPDF,visualDictionary)     
        descriptors.append(v)
        idPDF.append(sFirst)
                    
    #list to array    
    descriptors = np.asarray(descriptors)
    print("descriptors: {}".format(descriptors))
    print("idPDF: {}".format(idPDF))
    print("len descriptors : {}".format(descriptors.shape))
    print("len idpDF: {}".format(len(idPDF)))
    print("total number of PDF's: {}".format(docCont))
    print("processed number of PDF's: {}".format(docProcessed))

    return descriptors, idPDF


from sklearn.decomposition import PCA


def norm_feature_scale(X, x_min=0.0, x_max=1.0):
    old_shape = X.shape
    if len(old_shape) == 1:
        X = X.reshape(1, -1) 
        #X = np.expand_dims(X, axis=0)
    nom = (X.T-X.min(axis=1)).T*(x_max-x_min)
    denom = X.max(axis=1) - X.min(axis=1)
    denom[denom==0] = 1
    if len(old_shape) == 1:
        nom = nom[0]
    return x_min + (nom.T/denom).T

def norm_simple(X):
    row_maxs = X.max(axis=1)
    X = X / row_maxs[:, np.newaxis]
    return X

# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

def VLAD(X,visualDictionary, perform_pca=True, train_only=False):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    
    #computing the differences
    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            V[i]=np.sum((X[predictedLabels==i,:]-centers[i]),axis=0) 
    
    if type(perform_pca) == PCA and train_only:
        perform_pca.fit(V)
    elif type(perform_pca) == PCA and not train_only:
        V = perform_pca.transform(V)
    elif (type(perform_pca) == bool or type(perform_pca) == int) and perform_pca and not train_only:
        n_components = 16
        if type(perform_pca) is int:
            n_components = perform_pca // X.shape[1] # 16 components = 2048 (target dim) // 128 (sift shape)

        pca = PCA(n_components=n_components)
        V = pca.fit_transform(V)
    
    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    eps = 0.000001
    V = V/(np.sqrt(np.dot(V,V))+eps)
    
    return V



#Implementation of an improved version of VLAD
#reference: Revisiting the VLAD image representation
def improvedVLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V,V))
    return V

from sklearn.neighbors import DistanceMetric

def indexBallTree(X,leafSize):
    # custom distance metric for querying: 
    # this metric represents VLAD similarity
    def inverted_abs_dot(x,y):
        return 1 - np.abs(np.dot(x,y))
    dt=DistanceMetric.get_metric('pyfunc',func=inverted_abs_dot)
    tree = BallTree(X, leaf_size=leafSize) #, metric=dt)              
    return tree

#typeDescriptors =SURF, SIFT, OEB
#k = number of images to be retrieved
def query(image, k, descriptorName, visualDictionary,tree, perform_pca=False):
    #read image
    im=cv2.imread(image)
    #compute descriptors
    dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    funDescriptor=dict[descriptorName]
    kp, descriptor=funDescriptor(im)

    #compute VLAD
    v=VLAD(descriptor,visualDictionary, perform_pca=perform_pca)
    
    if len(v.shape) == 1:
        v = v.reshape(1, -1)  # we need to reshape like this if v contains single sample

    #find the k most relevant images
    dist, ind = tree.query(v, k, dualtree=True, breadth_first=True)    

    dist = dist.flatten()
    ind = ind.flatten()

    return dist, ind

# perform query with custom funDescriptor
def queryOnPatch(image_patch, k, funDescriptor, visualDictionary,tree):
    #read image
    #im=cv2.imread(image)
    #compute descriptors
    #dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    #funDescriptor=dict[descriptorName]
    kp, descriptor=funDescriptor(image_patch)

    #compute VLAD
    v=VLAD(descriptor,visualDictionary)

    if len(v.shape) == 1:
        v = v.reshape(1, -1)  # we need to reshape like this since v contains single sample

    #find the k most relevant images
    dist, ind = tree.query(v, k)    

    dist = dist.flatten()
    ind = ind.flatten()

    return dist, ind

def queryWithFun(img_path, k, funDescriptor, visualDictionary, tree, perform_pca=False):
    im=cv2.imread(img_path)
    #compute descriptors
    #dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    #funDescriptor=#dict[descriptorName]
    kp, descriptor=funDescriptor(im)
    print("DES SHAPE: ", descriptor.shape)
    #compute VLAD
    v=VLAD(descriptor,visualDictionary, perform_pca=perform_pca)

    if len(v.shape) == 1:
        v = v.reshape(1, -1)  # we need to reshape like this since v contains single sample

    #find the k most relevant images
    dist, ind = tree.query(v, k)    

    dist = dist.flatten()
    ind = ind.flatten()

    return dist, ind

def queryOnSIFT(descriptor, k, visualDictionary,tree):
    #read image
    #im=cv2.imread(image)
    #compute descriptors
    #dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    #funDescriptor=dict[descriptorName]
    #kp, descriptor=funDescriptor(image_patch)

    #compute VLAD
    v=VLAD(descriptor,visualDictionary)

    if len(v.shape) == 1:
        v = v.reshape(1, -1)  # we need to reshape like this since v contains single sample

    #find the k most relevant images
    dist, ind = tree.query(v, k)    

    return dist, ind


def queryOnVLAD(v, k, visualDictionary,tree):
    #read image
    #im=cv2.imread(image)
    #compute descriptors
    #dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    #funDescriptor=dict[descriptorName]
    #kp, descriptor=funDescriptor(image_patch)

    #compute VLAD
    #v=VLAD(descriptor,visualDictionary)

    if len(v.shape) == 1:
        v = v.reshape(1, -1)  # we need to reshape like this since v contains single sample

    #find the k most relevant images
    dist, ind = tree.query(v, k, dualtree=True, breadth_first=True) #, dualtree=True, breadth_first=True) #, breadth_first=True    
    #dist, ind = tree.query_radius(v, k, count_only=False, return_distance=True) #, dualtree=True, breadth_first=True) #, breadth_first=True    
    
    '''
    ind = tree.query(v, k, return_distance=False, dualtree=True)
    dist = None
    '''
    dist = dist.flatten()
    ind = ind.flatten()

    return dist, ind


def find_and_sort(query_desc, ref_desc, k, query_ids=None, ref_ids=None):
    if len(query_desc.shape) == 1:
        query_desc = query_desc.reshape((1, -1))
    
        
    #print("query, ref: ", query_desc.shape, ref_desc.shape)
    #for i in range(numQ):
    #    distMat[i, :] = np.linalg.norm(qLoc[i, :] - dbLoc, axis = 1)
    #distances = np.linalg.norm(query_desc - ref_desc.T, axis = 1)
    
    #print("query shape: ", query_desc.shape[0])
    distances = np.zeros((query_desc.shape[0], ref_desc.shape[0]))
    for ix in range(0, query_desc.shape[0]):
        #print("diff: ", (ref_desc - query_desc[ix].T).shape)
        #print("norm: ", np.linalg.norm(ref_desc - query_desc[ix].T, axis = 0).shape)
        distances[ix, :] = np.linalg.norm(ref_desc - query_desc[ix].T, axis = 1)
    print(distances.max(), distances.min(), np.mean(distances[:]))
    #input('?')
    #distances = np.expand_dims(distances, axis=0)
    
    #distances = np.matmul(query_desc, ref_desc.T)
    #distances = np.abs(distances)
    

    #print("Distances matmul shape: ", distances.shape, distances.min(), distances.max())
    if query_ids is None or ref_ids is None:
        indices = np.tile(np.arange(ref_desc.shape[0], dtype=int), [query_desc.shape[0], 1])
    else:
        
        
        # TODO: filter same indices with min distance:
        uniq_r = np.unique(ref_ids)
        uniq_q = np.unique(query_ids)
        
        uniq_d = np.zeros((uniq_q.shape[0], uniq_r.shape[0]))
        print("D shape: ", uniq_d.shape)
        for rix in uniq_r:
            #print("uniq: ", ix)
            for qix in uniq_q:
                print(distances[(query_ids == qix), 0].shape, distances[:,(ref_ids == rix)].shape)
                d = distances[query_ids == qix][ref_ids == rix].min()
                print(d.shape, d)
                uniq_d[qix, rix] = distances[query_ids == qix, ref_ids == rix].min()
           
        distances, indices = uniq_d, uniq
        print("uniq shape: ", uniq.shape, uniq_d.shape)
        input('?')
        #indices, unique_indices = np.unique(a, return_inverse=True)
    #print("Indices repeat shape: ", indices.shape)
    #input("Continue?")

    ''' # slower version of the above code (not vectorized)
    distances = np.zeros((query_desc.shape[0], ref_desc.shape[0]))
    indices = np.zeros((query_desc.shape[0], ref_desc.shape[0]), dtype=int)
    print(distances.shape, query_desc.shape,  ref_desc.shape)
    for qi in range(query_desc.shape[0]):
        q = query_desc[qi]
        for ri in range(ref_desc.shape[0]):
            r = ref_desc[ri]
            d = dist_fun(q, r)
            distances[qi, ri] = d
            indices[qi, ri] = ri
    '''
    assert distances.shape == indices.shape

    distances = distances.flatten()
    indices = indices.flatten()
    
    if False: #query_ids is not None and ref_ids is not None: 
        
        # remap the patch indices to image indices
        # then take unique indices with best similarities
        print("patch2imageNP!!!")

        for ix in range(patch2imageNP.shape[0]):
            mini, maxi, im_id = patch2imageNP[ix]
            #print("mini maxi im id: ", mini, maxi, im_id)
            mask = np.logical_and(mini <= indices, indices <= maxi)
            #print("tmp shape: ", mask.shape)
            indices[mask] = im_id

        #print("after: ", indices[1000:1050])
        #input("continue?")
        '''
        image_indices = np.empty(indices.shape)
        for ix in range(indices.shape[0]):
            i = indices[ix]
            for mini, maxi in patch2image.keys():
                if i >= mini and i <= maxi:
                    i = idImages.index(patch2image[(mini, maxi)])
                    image_indices[ix] = i
                    break
        '''
        # filter unique ids with best similarities (smallest distances)
        unique_ind = list(patch2imageNP[:,2])
        unique_dist = {}
        for ix in unique_ind:
            try:
                unique_dist[ix] = np.min(distances[ix == indices])
            except:
                pass
                #unique_ind.remove(ix)
        
        indices, distances = zip(*unique_dist.items())
        indices = np.array(indices)
        distances = np.array(distances)
        distances = distances / distances.max()
        #print("distances shape: ", distances.shape)
        #print("indices shape: ", indices.shape)
        #input("continue?")

    # sort from min to max distances
    p = distances.argsort() # sort from min to max distances
    distances = distances[p] # obtain ordered similarities
    #print("d[:10]: ", distances[:5], " \nd[:-10]: ", distances[-5:])
    indices = indices[p] # obtain ordered indices

    distances = distances[:k] # return k best results
    indices = indices[:k]
    
    return distances, indices


	




