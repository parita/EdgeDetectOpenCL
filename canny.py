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
from scipy import signal, ndimage
from opencl_kernels import *

class GPU():
    def __init__(self):
        # Select the desired OpenCL platform; you shouldn't need to change this:
        NAME = 'AMD Accelerated Parallel Processing'
	    # NAME = 'NVIDIA CUDA'	
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

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
        self.sobelx=np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],dtype=np.float32)
        self.sobely=np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]],dtype=np.float32)	
        self.gradientFunc = gpu.buildKernel(gradient_kernel).func
        self.gradientFunc.set_scalar_arg_dtypes([None, None, None, None, None, np.uint32, np.uint32, np.uint32, 
                                    None, None, None, np.uint32])
        self.thresholdFunc = gpu.buildKernel(threshold_kernel).func
        self.thresholdFunc.set_scalar_arg_dtypes([None, None, None, np.float32, np.uint32, np.uint32])

    def makeBorder(self, src_img):
        src_img = np.float32(src_img) / np.amax(np.amax(src_img))
        k = 3
        return copyMakeBorder(src_img,k/2,k/2,k/2,k/2, BORDER_CONSTANT, value = 0)

    def gradientxy(self, gpu, src_img, out_img, thres):
        k =3
        ckernelx_gpu = cl.array.to_device(gpu.queue, self.sobelx)
        ckernely_gpu = cl.array.to_device(gpu.queue, self.sobely)
        src_img_gpu = cl.array.to_device(gpu.queue, src_img)
        out_img_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        out_angles_gpu = cl.array.empty(gpu.queue, out_img.shape, src_img.dtype)
        ckernel_blockx_gpu = cl.LocalMemory(np.float32().nbytes*k*k)
        ckernel_blocky_gpu = cl.LocalMemory(np.float32().nbytes*k*k)
        img_block_gpu = cl.LocalMemory(np.float32().nbytes * (self.BLOCK_DIM + k) * (self.BLOCK_DIM + k))
        local = ((self.BLOCK_DIM), (self.BLOCK_DIM))
        [h,w] = out_img.shape
        evt = self.gradientFunc(gpu.queue, out_img.shape, local, 
                            ckernelx_gpu.data, ckernely_gpu.data, src_img_gpu.data,
                            out_img_gpu.data, out_angles_gpu.data, k, h, w, 
                            ckernel_blockx_gpu, ckernel_blocky_gpu, 
                            img_block_gpu, self.BLOCK_DIM)
        evt.wait()
        t = 1e-9 * (evt.profile.end - evt.profile.start)
        """
        angles = abs(np.arctan(Gy/Gx)) * 4 / np.pi
        angles = np.int32(angles)
        """
        src_img = out_img_gpu.get()
        src_img_gpu = cl.array.to_device(gpu.queue, src_img)
        evt = self.thresholdFunc(gpu.queue, out_img.shape, None, src_img_gpu.data, out_img_gpu.data, out_angles_gpu.data, thres, h, w)
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
        #out_img = out_img / np.amax(out_img)
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

    def Gradient(self, src_img, thres):
        BLOCK_DIM = 32
        Gxy = GradientClass(BLOCK_DIM)
        [old_dim, src_img] = self.adjust_image_size(src_img, BLOCK_DIM)
        # For blurring, we convolve with the guassian kernel
        out_img = np.empty(src_img.shape, dtype = src_img.dtype)
        src_img = Gxy.makeBorder(src_img)
        [t, out_img] = Gxy.gradientxy(self.gpu, src_img, out_img, thres)
        out_img = self.adjust_output(out_img, old_dim)
        return t, out_img

    def openclCanny(self, src_img, thres, k = 25):
        [t, out_img] = self.blur(src_img, k)
        [t1, out_img] = self.Gradient(out_img, thres)
        return [t+t1, out_img]

    def pythonCanny(self, src_img, k = 15):
        BLOCK_DIM = 32
        convolve_class = ConvolveClass(k, BLOCK_DIM)
        kernel = convolve_class.guassian(k)
        kernel = kernel / np.amax(kernel)
        start = time.time()
        out_py = signal.convolve2d(src_img, kernel, mode='same', boundary='fill', fillvalue=0)
        Gx = ndimage.filters.sobel(out_py, axis=0, output=None, mode='constant', cval=0.0)
        Gy = ndimage.filters.sobel(out_py, axis=1, output=None, mode='constant', cval=0.0)
        out_py = np.sqrt(Gx*Gx + Gy*Gy)
        angles = abs(np.arctan(Gy/Gx)) * 4 / np.pi
        idx = out_img[:,:] > thres
        out_img[idx] = 1.0
        idx = out_img[:,:] <= thres
        out_img[idx] = 0.0
        t = time.time() - start
        out_py = np.array(out_py)
        out_py = out_py / np.amax(np.amax(out_py))
        return t, out_py

class plot():
    def __init__(self):
        pass

    def plot_vs_size(self, X, func1, func2, func3):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(X, func1, 'ro-',
                 X, func2, 'bo-')
        plt.xlabel('Size of input - k x k')
        plt.ylabel('Time measurement')
        plt.title('Canny Edge detector')
        plt.legend(('OpenCL', 'Python'), loc = 'upper left')
        plt.grid(True)
        #plt.gca().set_xlim((min(X), max(X)))  
        plt.draw()
        plt.savefig('Canny Time Measurement.png')
        
        with open('Canny Time Measurement.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['Size', 'OpenCL Time', 'Python Time'])
            for x, t1, t2 in zip(X, func1, func2):
                w.writerow([x, t1, t2])
            w.writerow(['Average time per video frame:', sum(func3)/len(func3)])

if __name__ == "__main__":
    cap = VideoCapture(0)
    gpu = GPU()
    canny = Canny(gpu)
    src_img = imread('Lenna.png', 0)
    thres = 0.3
    k = src_img.shape[0]
    k_list = []
    time_cl = []
    time_py = []
    while k < 1500:
        k_list.append(k)
        src_img = resize(src_img, (k, k))
        
        [t, out_img] = canny.openclCanny(src_img, thres, 25)
        imshow("Output", out_img)
        time_cl.append(t)
        print "OpenCL Time:", t

        [t, out_py] = canny.pythonCanny(src_img, 25)
        time_py.append(t)
        imshow("Python Output", out_py)
        print "Python Time:" , t
        
        k = k + 200
        if waitKey(1) & 0xFF == ord('q'):
            break
   
    time_video = []
    i = 0
    thres = 0.1
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        src_img = cvtColor(frame, COLOR_BGR2GRAY)

        # OpenCL Convolution
        [t, out_img] = canny.openclCanny(src_img, thres, 25)
        time_video.append(t)
        imshow("Output", out_img)
        print "OpenCL Time:", t

        # Python Convolution
        #[t, out_py] = canny.pythonCanny(src_img, 25)
        #start = time.time()
        #out_py = Canny(src_img)
        #print "Python Time:", time.time() - start
        #print out_py.shape

        imshow('input', src_img)
        out_img = out_img / np.amax(out_img)
        out_img = out_img * 255.0
        imwrite('output' + str(i) + '.png', out_img)    
        i = i + 1
        if waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    destroyAllWindows()
    
    p = plot()
    p.plot_vs_size(k_list, time_cl, time_py, time_video)
