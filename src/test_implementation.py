#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:10:45 2020

@author: iason
"""


import numpy as np
import cv2 as cv
import os
import sys
from sklearn.cluster import KMeans

# Windows specific path
if sys.platform.startswith('win32'):
    init_path = "D:\Projects\MOTSChallenge2020\dataset\MOTSChallenge\\train\images\\"

# Linux specific path
elif sys.platform.startswith('linux'):
    init_path = '/home/iason/Projects/MOTSChallenge2020/dataset/MOTSChallenge/train/images/'

# Sub-directories tuple
# TODO: implement non-stationary sequencies
dirs = ('0009')

# Loop through each sub-directory
# for i in dirs:
# Select current sub-directory
path = os.path.join(init_path, dirs)



try:
    os.chdir(path)
# Error handling
except  OSError:
    sys.exit("Invalid path.")

# Initialize frame stream source
cap = cv.VideoCapture('%6d.jpg')

# 9 by 9 rectangular window for morph operators
kernel = np.ones((9, 9), dtype=np.uint8) 
# Zivkovic MOG
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = False)

# contours px size to accept
contour_size = 60

# Frame shape flag
flg_shp = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame valid
    if not ret:
        break
    
    # Evaluate frame's shape
    if flg_shp:
        height, width, col = np.shape(frame)
        
        # Set flag status to false
        flg_shp = False
    
    # Evaluate foreground mask
    fgmask = fgbg.apply(frame)
    
    # Image processing function
    # Self bitwise operation on current frame
    res = cv.bitwise_and(frame,frame, mask= fgmask)
    
    # Morphologically dilate current frame
    e_im = cv.dilate(fgmask, kernel, iterations = 1)
    
    # Morphologically close current frame
    e_im = cv.morphologyEx(e_im, cv.MORPH_CLOSE, kernel)
    
    # Evaluate & capture each entire silhouettes
    contours, hierarchy = cv.findContours(e_im, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    
    # Remove contours that are lower than threshold's value
    temp = []
    num = 0
    # Loop through each detected contours
    for i in range(len(contours)):
        # If current contour size less than threshold's value, store contour
        if len(contours[i]) < contour_size:
            temp.append(i)
    # Loop through each contour that is less than threshold's value
    for i in temp:
        # Delete the contours that are less than threshold's value
        del contours[i - num]
        num = num + 1
    
    # Loop though each contour's mask
    for i in range(len(contours)):
        # Select current contour
        temp_cnt = contours[i]
        
        # Initialize mask
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Evaluate mask of current contour
        temp_mask = cv.drawContours(temp_mask, [temp_cnt], 0, 255, -1)
        
        # Isolate selected contour in current frame
        tmp_frame = cv.bitwise_and(frame, frame, mask=temp_mask)
        
        # Approach 1: K-means on histogram
        # Initialize histogram array of each colour channel
        hist = np.zeros((255, 3), dtype=np.uint32)
        for c in range(col):
            # Evaluate histogram; 255 due to Red contour draw(Red channel only.
            hist[:, c] = np.reshape(cv.calcHist([frame], [c], temp_mask, [255], [0, 255]),
                                    (255))
            
            # # Apporach 1.1: Find optimal number of clusters
            # # TODO: Implement cluster estimator; for loop & array sizes.
            # # Initialize squared distances
            # dist = np.zeros()
            # # Normalized squared distances
            # dist_norm = np.zeros()
            # # K-Means flag
            # k_flg = False
            # # Optimal k clusters
            # for i in range():
            #     # Optimal number of clusters
            #     if not k_flg:
            #         # Implement K-Means model
            #         kmeans = KMeans(n_clusters=i+1, init='k-means++', n_jobs=8)
            #     else:
            #         kmeans = KMeans(n_clusters=i, init='k-means++', n_jobs=8)
                
            #     # Fit input histogram
            #     kmeans.fit(hist[:, c])
                
            #     # Break if optimal k found
            #     if k_flg:
            #         # Cluster current histogram
            #         labels = kmeans.predict(np.reshape(hist[:, c], (255, 1)))
            #         break
                
            #     # Store squared distances results
            #     dist[i] = kmeans.inertia_
                
            #     # Alt: sklearn.preprocessing normalize; norm(x[:i+1], axis=0).ravel().
            #     # Omits initialization of array. Re-initializes after each iteration (bad?)
            #     # OR x/np.linalg.norm(x)
                # if i>0:
                    # dist_norm[i-1] = (dist[:i+1] - np.min(dist[:i+1])) / (np.max(dist[:i+1]
            #                                                       - np.min(dist[:i+1])))
            #     # Determine optimal number of clusters
            #     if dist_norm[i-1] < 0.1 and dist_norm[i-1] > 0:
            #         # Set k-means flag to True
            #         k_flg = True
            # # Approach 1.2: Cluster estimator approachalt
            # kmeans = KMeans(n_clusters, init='k-means++', n_jobs=8)
            
            # # Fit current histogram
            # kmeans.fit(hist[:, c])
            
            # # Cluster current histogram
            # labels = kmeans.predict(hist[:, c])
            
        # Approach 2: K-Means on isolated contours
        # Could either utilzie cluster estimator or optimal k
        # res_cnt = kmeans_cv(tmp_frame, n_clusters)
    # Perform bitwise and operation using the morphological processed frame as
    # a mask
    res2 = cv.bitwise_and(frame,frame, mask = e_im)
    
    
    # Show resulting frames
    # cv.imshow('Foreground', res2)
    cv.drawContours(res, contours, -1, (0, 0, 255), 2)
    cv.imshow('Contours', res)
    
    # To break, press the q key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release cap & close all windows
cap.release()
cv.destroyAllWindows()


def imBackSub(path):
    '''
    
    
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.
    
    '''
    
    def BackSub(frame, bg_frame, thresh):
        '''
        
        
        Parameters
        ----------
        frame : TYPE
            DESCRIPTION.
        bg_frame : TYPE
            DESCRIPTION.
        thresh : TYPE
            DESCRIPTION.
        
        Returns
        -------
        TYPE
            DESCRIPTION.
        
        '''
        
            # Assert if RGB
        try:
            # Evaluate dimensions
            height, width, col = np.shape(frame)
        # Frame is Grayscale
        except ValueError:
            # Evaluate dimensions
            height, width = np.shape(frame)
        
        # Evaluate absolute difference between frames
        fr_diff = np.abs(np.int16(np.subtract(frame, bg_frame)))
        
        # Initialize output frame array
        fg = np.zeros_like(frame, dtype=np.uint8)
        
        # Threshold background subtraction
        fg = np.where(fr_diff > thresh, np.uint8(frame), np.uint8(0))
        
        # INFO: Slow performance with the code below
        # # Loop though frame's dimensions
        # for c in range(col):
        #     for i in range(width):
        #         for k in range(height):
        #             # If current pxl value > thresh, then pxl foreground
        #             if (fr_diff[k, i, c] > thresh):
        #                 fg[k, i, c] = frame[k, i, c]
        #             # Set pxl value to zero
        #             else:
        #                 fg[k, i, c] = np.uint8(0)
        # Set current frame as background frame for next iteration
        bg_frame = frame
        
        # Return foreground & updated background
        return (np.array([fg, bg_frame]))
    
    # Change directory
    try:
       os.chdir(path)
    # Error handling
    except  OSError:
        sys.exit("Invalid path.")
    
    # Initialize frame stream source
    cap = cv.VideoCapture('%6d.jpg')
    
    # Read first frame
    bg_frame = cv.imread('000001.jpg')
    
    # Background subtraction threshold
    thresh = 250
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame valid
        if not ret:
            break
        
        (res2, bg_frame) = BackSub(frame, bg_frame, thresh)
        
        # Show resulting frames
        cv.imshow('Foreground', res2)
        
        # To break, press the q key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # Release cap & close all windows
    cap.release()
    cv.destroyAllWindows()





# Implement proposed solution
# main_f(path)
# imBackSub(path)
