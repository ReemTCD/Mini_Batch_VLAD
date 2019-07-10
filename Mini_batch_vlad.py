"""
VLAD encoding of SIFT using Mini Batch k-means clustering.
"""

# using implementation from  Jorge Guevara; jorjasso: https://github.com/jorjasso/VLAD
from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2
import os

from utils import read_files_recursive
from sift import detect_sift



    #all_descriptors = np.zeros((0, 128), dtype=np.int16) # for SIFT:
    all_descriptors = np.zeros((0, 128), dtype=np.int16) # for SIFT:
    list_descriptors = []
    filtered_descriptors = []
    filtered_keypoints = []
    cntr = 0
    #for coord in coordinates:
    #    bbox = cv2.boundingRect(coord)   
    for bbox in reduced_bboxes:
        x,y,w,h = bbox
        
        if w< 10 or h < 10 or w > gray.shape[1]/2. or h > gray.shape[0]/2.:  # w/h > 5 or h/w > 5:
            continue

        x,y,w,h = resize_patch(x,y,w,h, gray.shape[0], gray.shape[1])
        patch = gray[y:y+h, x:x+w]

         #print("ORIG shape: ", gray.shape, gray.max(), gray.dtype)
        h,w = patch.shape
        if h !=  w:
            #cv2.imwrite("demo/orig/orig-patch_{}_{}.png".format(offset_x, offset_y), gray)
            # normalize gray region to square before detecting SIFT
            patch = cv2.resize(patch, None, fy=1. if h < w else 1.* w/h, fx=1. if h > w else 1.*h/w)
            #print("Squarified shape: ", clr_patch.shape, clr_patch.max(), clr_patch.dtype)
            #cv2.imwrite("demo/squ/squared-patch_{}_{}.png".format(offset_x, offset_y), cv2.resize(gray, (128,128)))
        if save_dir is not None and label is not None and cntr % 15 == 0:
            clr_patch = img[y:y+h, x:x+w, :]
            cv2.imwrite(os.path.join(save_dir, "{}_squared-patch_{}_{}_{}_{}.png".format(label, x, y, w, h)), 
                cv2.resize(clr_patch, (128,128)))

        kp, des = detect_sift(patch, offset_x=x, offset_y=y)
        
        if type(des) is np.ndarray:  
            list_descriptors.append(des)
            '''
            j = all_descriptors.shape[0]
            max_desc = all_descriptors.shape[0] + des.shape[0]
            all_descriptors.resize((max_desc, des.shape[1]), refcheck=True) # works)
            all_descriptors[j:j+des.shape[0],:] = des.copy()
            del max_desc
            del des
            '''
         
        cntr += 1
    
    if _debug:
        print("Total detected keypoints: ", len(filtered_keypoints))

   

    return filtered_keypoints, list_descriptors #all_descriptors # filtered_keypoints

# TODO: for debug purposes and visualising positive / negative features
def drawlines(img, points):
    filler = cv2.convexHull(points)
    cv2.polylines(img, filler, True, (0, 0, 0), thickness=2)
    return img 

if __name__ == '__main__':

    do_test = True
    if do_test:
        demo_dir = 'demo/'

        img = cv2.imread(demo_dir+'test1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = extract_sift_within_mser(gray, _debug=True)    

        #print("KP outside: ", len(kp))
        #print("DESC outside: ", len(des))

        img=cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(demo_dir+'bundled_sift_keypoints.jpg', img)

        path = 'demo/'
        
        import time
        start = time.time()
        descriptors = getDescriptors(path, extract_sift_within_mser)
        end = time.time()
        print("Total time [s]: ", end - start)
        # around 8[s] 
        print("Avg. time per image [s]: ", (end - start) / (float)(len(read_files_recursive(path, True, False))))

        #writting the output
        save_to = "pickles/vlad_descriptors.pickle"
        with open(save_to, 'wb') as f:
            pickle.dump(descriptors, f)

    else:
        path = 'data/images/'
        parts = os.listdir(path)

        # import sys
        from tqdm import tqdm
        for part in tqdm(parts):
            p = os.path.join(path, part)

            # prevent printing on stdout
            #sys.stdout = open(os.devnull, "w")
            
           

            #writting the output
            save_to = "pickles/vlad_{}.pickle".format(part)

            with open(save_to, 'wb') as f:
                pickle.dump(descriptors, f)

