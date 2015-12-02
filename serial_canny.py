#!/usr/bin/env python

import time

import numpy as np
from cv2 import * 

max_thres = 500
ratio = 5

def CannyThreshold(lowThreshold):
    start = time.time()
    out_img = Canny(src_img, lowThreshold, lowThreshold*ratio)
    runtime = time.time() - start
    print "Running time:", runtime
    global time_list
    time_list.append(runtime)
    imshow('Output', out_img)

src_img = imread('Lena.bmp', 0)
out_img = src_img

namedWindow('Output')
time_list = []
print createTrackbar('lowThreshold', 'Output', 0, max_thres, CannyThreshold)
imshow('Output', out_img)
while (True):
    if waitKey() & 0xFF == 27:
        break
if len(time_list) > 0:
    print "Average running time: ", sum(time_list)/len(time_list)
destroyAllWindows()



