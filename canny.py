#!/usr/bin/env python

"""
Rectangular matrix multiplication using PyOpenCL.
"""

import csv
import time

import pyopencl as cl
import pyopencl.array
import numpy as np
from cv2 import * 
from scipy import signal
from opencl_kernels import *

class GPU():
    def __init__(self):
        # Select the desired OpenCL platform; you shouldn't need to change this:
        NAME = 'AMD Accelerated Parallel Processing'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
                #for device in devs:
                #    print("Device max work group size:", device.max_work_group_size)
                #    print("Device max work item sizes:", device.max_work_item_sizes)

        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx)
    
    def buildKernel(self, kernel):
        return cl.Program(self.ctx, kernel).build().func

       
class ConvolveClass():
    def __init__(self, k = 15, BLOCK_DIM = 32):
        self.BLOCK_DIM = BLOCK_DIM
        self.k = k
        self.sigma = 8*2*np.pi/self.k
        self.convolve = gpu.buildKernel(convolution_kernel)
        self.convolve.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32, 
                                    None, None, np.uint32])

    def setK(self, k):
        self.k = k
        self.sigma = 4*2*np.pi/self.k

    def setBLOCKDIM(self, b):
        self.BLOCK_DIM = b

    def guassian(self, k = None):
        if k:
            self.setK(k)
        kernel = getGaussianKernel(self.k, self.sigma) * np.transpose(getGaussianKernel(self.k, self.sigma))
        return np.float32(kernel)
        
    def makeBorder(self, src_img):
        src_img = np.float32(src_img) / np.amax(np.amax(src_img))
        return copyMakeBorder(src_img,self.k/2,self.k/2,self.k/2,self.k/2, BORDER_CONSTANT, value = 0)

    def convolution(self, gpu, src_img, out_img, kernel):
        ckernel_gpu = cl.array.to_device(gpu.queue, kernel)
        src_img_gpu = cl.array.to_device(gpu.queue, src_img)
        out_img_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        ckernel_block_gpu = cl.LocalMemory(np.float32().nbytes*self.k*self.k)
        img_block_gpu = cl.LocalMemory(np.float32().nbytes * (self.BLOCK_DIM+self.k) * (self.BLOCK_DIM+self.k))
        local = ((self.BLOCK_DIM), (self.BLOCK_DIM))
        [h,w] = out_img.shape
        self.convolve(gpu.queue, out_img.shape, local, ckernel_gpu.data, src_img_gpu.data, out_img_gpu.data, self.k, h, w, ckernel_block_gpu, img_block_gpu, self.BLOCK_DIM)
        return out_img_gpu.get()

class Canny():
    def __init__(self, gpu):
        self.convolve_class = ConvolveClass()
        self.k = self.convolve_class.k
        self.gpu = gpu
        self.kernel = None

    def blur(self, src_img, k):
        self.k = k
        [h,w] = src_img.shape
        [old_h, old_w] = [h,w]
        src_img = np.float32(src_img)
        if (h % 32 != 0):
            old_h = h
            h = (np.int32(h/32) + 1) * 32
        if (w % 32 != 0):
            old_w = w
            w = (np.int32(w/32) + 1) * 32
        src_img = resize(src_img, (h,w))
        self.kernel = self.convolve_class.guassian(self.k)
        out_img = np.empty(src_img.shape, dtype = src_img.dtype)
        src_img = self.convolve_class.makeBorder(src_img)
        out_img = self.convolve_class.convolution(self.gpu, src_img, out_img, self.kernel)
        out_img = out_img / np.amax(out_img)
        out_img = resize(out_img, (old_w,old_h))
        return out_img

    def pythonBlur(self, src_img, k = 15):
        if self.kernel == None:
            self.kernel = self.convolve_class.guassian(k)
            self.kernel = self.kernel / np.amax(kernel)
        out_py = signal.convolve2d(src_img, self.kernel, mode='same', boundary='fill', fillvalue=0)
        out_py = np.array(out_py)
        out_py = out_py / np.amax(np.amax(out_py))
        return out_py

if __name__ == "__main__":
    gpu = GPU()
    src_img = imread('Lenna.png', 0)
    #src_img = resize(src_img, (1502, 1502))
    # Add function that adjusts the size and sets/changes the block dimennsion if necessary
    canny = Canny(gpu)

    # OpenCL Convolution
    out_img = canny.blur(src_img, 25)

    # Python Convolution
    out_py = canny.pythonBlur(src_img, 25)
    print out_py.shape

    print "Comparison with python:", np.allclose(out_py, out_img)
    #print "Python Output:", out_py
    #print "OpenCL Output:", out_img

    imshow('input', src_img)
    imshow('output', out_img)
    out_img = out_img * 255.0
    waitKey(0)
    destroyAllWindows()
    imwrite('output.png', out_img)
