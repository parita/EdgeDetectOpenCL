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
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    def buildKernel(self, kernel):
        return cl.Program(self.ctx, kernel).build()

       
class ConvolveClass():
    def __init__(self, k = 15, BLOCK_DIM = 32):
        self.BLOCK_DIM = BLOCK_DIM
        self.k = k
        self.sigma = 8*2*np.pi/self.k
        self.convolve_prg = gpu.buildKernel(convolution_kernel)
        self.convolve = self.convolve_prg.func
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
        
    def makeBorder(self, src_img, k = None):
        if k:
            self.setK(k)
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
        evt = self.convolve(gpu.queue, out_img.shape, local, ckernel_gpu.data, src_img_gpu.data, out_img_gpu.data, self.k, h, w, ckernel_block_gpu, img_block_gpu, self.BLOCK_DIM)
        evt.wait()
        t = 1e-9*(evt.profile.end - evt.profile.start)
        return t, out_img_gpu.get()

class GradientClass():
    def __init__(self, BLOCK_DIM = 32):
        self.BLOCK_DIM = BLOCK_DIM
        self.sobel = np.array([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]], dtype = np.float32)
        self.gradient_prg = gpu.buildKernel(gradient_kernel)
        self.gradientFunc = self.gradient_prg.func
        self.gradientFunc.set_scalar_arg_dtypes([None, None, None, None, None, np.uint32, np.uint32, np.uint32, 
                                    None, None, None, np.uint32])
        self.thresholdFunc = gpu.buildKernel(threshold_kernel).func
        self.thresholdFunc.set_scalar_arg_dtypes([None, None, np.float32, np.uint32, np.uint32])

    def makeBorder(self, src_img):
        src_img = np.float32(src_img) / np.amax(np.amax(src_img))
        k = 3
        return copyMakeBorder(src_img,k/2,k/2,k/2,k/2, BORDER_CONSTANT, value = 0)

    def gradientxy(self, gpu, src_img, out_img, thres):
        k =3
        ckernel_gpu = cl.array.to_device(gpu.queue, self.sobel)
        src_img_gpu = cl.array.to_device(gpu.queue, src_img)
        out_img_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        Gx_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        Gy_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        ckernel_blockx_gpu = cl.LocalMemory(np.float32().nbytes*3*3)
        ckernel_blocky_gpu = cl.LocalMemory(np.float32().nbytes*3*3)
        img_block_gpu = cl.LocalMemory(np.float32().nbytes * (self.BLOCK_DIM + k) * (self.BLOCK_DIM + k))
        local = ((self.BLOCK_DIM), (self.BLOCK_DIM))
        [h,w] = out_img.shape
        evt = self.gradientFunc(gpu.queue, out_img.shape, local, ckernel_gpu.data, src_img_gpu.data, Gx_gpu.data, Gy_gpu.data, 
                            out_img_gpu.data, k, h, w, ckernel_blockx_gpu, ckernel_blocky_gpu, img_block_gpu, self.BLOCK_DIM)
        evt.wait()
        t = 1e-9 * (evt.profile.end - evt.profile.start)
        src_img_gpu = cl.array.to_device(gpu.queue, out_img_gpu.get())
        evt = self.thresholdFunc(gpu.queue, out_img.shape, None, src_img_gpu.data, out_img_gpu.data, thres, h, w)
        evt.wait()
        t = t + 1e-9 * (evt.profile.end - evt.profile.start)
        return t, out_img_gpu.get()

class Canny():
    def __init__(self, gpu):
        self.gpu = gpu

    def adjust_image_size(self, src_img, BLOCK_DIM):
        # Adjust the size of the image for local dimension (BLOCK_DIM x BLOCK_DIM)
        [h,w] = src_img.shape
        [old_h, old_w] = [h,w]
        src_img = np.float32(src_img)
        if (h % BLOCK_DIM != 0):
            old_h = h
            h = (np.int32(h/BLOCK_DIM) + 1) * BLOCK_DIM
        if (w % BLOCK_DIM != 0):
            old_w = w
            w = (np.int32(w/BLOCK_DIM) + 1) * BLOCK_DIM
        src_img = resize(src_img, (h,w))
        return [(old_w, old_h), src_img]

    def adjust_output(self, out_img, (h, w)):
        out_img = out_img / np.amax(out_img)
        return resize(out_img, (h, w))
        
    def blur(self, src_img, k):
        BLOCK_DIM = 32
        convolve_class = ConvolveClass(k, BLOCK_DIM)
        [old_dim, src_img] = self.adjust_image_size(src_img, BLOCK_DIM)
        # For blurring, we convolve with the guassian kernel
        kernel = convolve_class.guassian(k)
        
        out_img = np.empty(src_img.shape, dtype = src_img.dtype)
        src_img = convolve_class.makeBorder(src_img)
        [t, out_img] = convolve_class.convolution(self.gpu, src_img, out_img, kernel)
        out_img = self.adjust_output(out_img, old_dim)
        return t, out_img

    def Gradient(self, src_img):
        BLOCK_DIM = 32
        Gxy = GradientClass(BLOCK_DIM)
        [old_dim, src_img] = self.adjust_image_size(src_img, BLOCK_DIM)
        # For blurring, we convolve with the guassian kernel
        out_img = np.empty(src_img.shape, dtype = src_img.dtype)
        src_img = Gxy.makeBorder(src_img)
        [t, out_img] = Gxy.gradientxy(self.gpu, src_img, out_img, 0.9)
        out_img = self.adjust_output(out_img, old_dim)
        return t, out_img

    def laplacianAlongGradient(self, src_img, Gx, Gy, k):
        BLOCK_DIM = k
        LoG = ConvolveClass(3, BLOCK_DIM)
        sobel_r = np.array([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]], dtype = np.float32)
        guassian_src = LoG.guassian(k)
        print guassian_src.shape
        out_img = np.empty(guassian_src.shape, dtype = guassian_src.dtype)
        img = LoG.makeBorder(guassian_src, 3)
        k_x = LoG.convolution(self.gpu, img, out_img, sobel_r)
        k_x = self.adjust_output(k_x, k_x.shape)
        sobel_c = np.transpose(sobel_r)
        k_y = LoG.convolution(self.gpu, img, out_img, sobel_c)
        k_y = self.adjust_output(k_y, k_y.shape)
        DoG_kernel = k_x + k_y  #np.sqrt(k_x*k_x + k_y*k_y)
        print DoG_kernel
        print k
        LoG = ConvolveClass(k, 32)
        out_img = np.empty(src_img.shape, dtype = src_img.dtype)
        src_img = LoG.makeBorder(src_img, k)
        out_img = LoG.convolution(self.gpu, src_img, out_img, DoG_kernel)
        out_img = self.adjust_output(out_img, out_img.shape)
        return out_img

    def pythonBlur(self, src_img, k = 15):
        BLOCK_DIM = 32
        convolve_class = ConvolveClass(k, BLOCK_DIM)
        kernel = convolve_class.guassian(k)
        kernel = kernel / np.amax(kernel)
        start = time.time()
        out_py = signal.convolve2d(src_img, kernel, mode='same', boundary='fill', fillvalue=0)
        t = time.time() - start
        out_py = np.array(out_py)
        out_py = out_py / np.amax(np.amax(out_py))
        return t, out_py

if __name__ == "__main__":
    cap = VideoCapture(0)
    gpu = GPU()
    canny = Canny(gpu)
    #fgbg = createBackgroundSubtractorMOG2()   
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #fgmask = fgbg.apply(frame)
        src_img = cvtColor(frame, COLOR_BGR2GRAY)
        #src_img =  src_img & fgmask
        imshow('frame', src_img)
        # Our operations on the frame come here
        #src_img = imread('Lenna.png', 0)
        #src_img = resize(src_img, (1502, 1502))
        # Add function that adjusts the size and sets/changes the block dimennsion if necessary

        # OpenCL Convolution
        start = time.time()
        [t, out_img] = canny.blur(src_img, 25)
        [t1, out_img] = canny.Gradient(src_img)
        #[t2, out_img] = canny.Gradient(out_img)
        #out_img = canny.laplacianAlongGradient(out_img, [], [], 25)
        print "Convolution Time:", t + t1
        print "OpenCL Time:", time.time() - start

        # Python Convolution
        #[t, out_py] = canny.pythonBlur(src_img, 25)
        #out_py = Canny(src_img)
        #print "Python Time:", t
        #print out_py.shape

        #print "Comparison with python:", np.allclose(out_py, out_img)
        #print "Python Output:", out_py
        #print "OpenCL Output:", out_img

        # Display the resulting frame
        #cv2.imshow('frame',gray)
        imshow('input', src_img)
        imshow('output', out_img)
        out_img = out_img * 255.0
        imwrite('output' + str(i) + '.png', out_img)    
        i = i + 1
        if waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    destroyAllWindows()
    
