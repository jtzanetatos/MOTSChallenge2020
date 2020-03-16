#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:53:03 2020

@author: iason
"""


import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morph
import cv2 as cv
import open3d as o3d

# Test frame's path
path = '0test_frame.jpg'

# Read test frame
img = cv.imread(path)

# Grayscale frame; can be considered as mask
grimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Skeletonize frame
skel, distance = morph.medial_axis(grimg, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

## Approach 1: Morphological opening

# # First stage - dilation
# res = morph.binary_dilation(dist_on_skel, selem=morph.diamond(2))

# # Two stages of erosion
# res = morph.binary_erosion(res, selem=np.ones((4, 4)))

# # Final skeleton model
# res = morph.binary_erosion(res, selem=np.ones((2, 2)))

# ## Approach 2: Histogram on masked frame
# hist = cv.calcHist([grimg], [0], None, [253], [3,255])

## Approach 3: Local maxima

# Evaluate local maxima
# l_max = morph.local_maxima(distance, connectivity=2)

# # Dilate local maxima mask
# l_max = np.uint8(morph.binary_dilation(l_max, selem=morph.disk(4)))

# # Evaluate mask
# l_mask = cv.bitwise_and(np.uint8(skel), cv.bitwise_xor(np.uint8(skel), l_max))

# # Apply mask
# skel_frame = cv.bitwise_and(img, img, mask=l_mask)

# ## Approach 4: Global maxima
# g_max = np.max(dist_on_skel)

## Approach 6: Normalize distance &apply Low Pass FIR Filter

# Normalize distance; min values can be omited
dist_norm = (distance - distance.min()) / (distance.max() - distance.min())

# Apply theshold/ FIR operation
dist_norm = np.where(dist_norm >=0.8, np.float32(0), dist_norm)

# Evaluate mask
skel_mask = np.uint8((dist_norm * 100) * skel)

# Apply mask on current frame
skel_frame = cv.bitwise_and(img, img, mask=skel_mask)