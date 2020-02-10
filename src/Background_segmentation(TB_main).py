#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:16:38 2020

@author: iason
"""

import numpy as np
import cv2 as cv
import os
import sys
import my_functions as f

__author__ = ""
__copyright__ = ""
__license__ = ""
__version__ = ""
__email__ = "TBA"


def main_f(path):
    '''
    
    
    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    
    Returns
    -------
    None.
    
    '''
    
    # Change directory
    try:
        w_dir = os.chdir(path)
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
    
    # Initialize frame array
    res2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame valid
        if not ret:
            break
        
        # Evaluate foreground mask
        fgmask = fgbg.apply(frame)
        
        # Image processing function
        (res2, res, contours) = f.frame_proc(frame, fgmask, kernel, contour_size)
        
        
        # Show resulting frames
        cv.imshow('Foreground', res2)
        
        
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
        
        fg = np.where(fr_diff > thresh, frame, np.uint8(0))
        
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
        
        return (np.array([fg, bg_frame]))
    
    # Change directory
    try:
        w_dir = os.chdir(path)
    # Error handling
    except  OSError:
        sys.exit("Invalid path.")
    
    # Initialize frame stream source
    cap = cv.VideoCapture('%6d.jpg')
    
    # Read first frame
    bg_frame = cv.imread('000001.jpg')
    
    # Background subtraction threshold
    thresh = 20
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if frame valid
        if not ret:
            break
        
        (res2, bg_frame) = BackSub(frame, bg_frame, thresh)
        i = 1
        # Show resulting frames
        cv.imshow('Foreground', res2)
        
        # To break, press the q key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # Release cap & close all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Windows specific path
    if sys.platform.startswith('win32'):
        # Image directory
        init_path = "D:\Projects\MOTSChallenge2020\dataset\MOTSChallenge\\train\images\\"
    
    # Linux specific path
    else:
        init_path = '/home/iason/Projects/MOTSChallenge2020/dataset/MOTSChallenge/train/images/'
    
    # Sub-directories tuple
    # TODO: implement non-stationary sequencies
    dirs = ('0002', '0009')
    
    # Loop through each sub-directory
    for i in dirs:
        # Select current sub-directory
        path = os.path.join(init_path, i)
        
        # Implement proposed solution
        main_f(path)
        # imBackSub(path)
    