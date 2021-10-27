#!/usr/bin/env python3

import cv2, pyfakewebcam, time, argparse, mediapipe, traceback
import numpy as np
import tkinter as tk

import code

'''
from inotify_simple import INotify, flags
import itertools
import signal
import sys
import configargparse
from functools import partial
from typing import Any, Dict
import os
import fnmatch
from cmapy import cmap
'''

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--webcam-path-real', dest='cam_path_real', default="/dev/video0", help="Set real webcam path")
parser.add_argument('-f', '--webcam-path-fake', dest='cam_path_fake', default="/dev/video2", help="V4l2loopback device path")

parser.add_argument('-W', '--width', default=640, type=int, help="Resolution width.")
parser.add_argument('-H', '--height', default=480, type=int, help="Resolution height.")
parser.add_argument('-F', '--framerate', default=24, type=int, help="Framerate.")

#parser.add_argument('--background-blur', dest='bg_blur_kernel', default=21, type=int, metavar='k', help="The gaussian bluring kernel size in pixels. MUST BE AN ODD NUMBER. Zero disables blurring.")
#parser.add_argument("--background-blur-sigma-frac", dest='bg_blur_sigma', default=3, type=int, metavar='frac', help="The fraction of the kernel size to use for the sigma value (ie. sigma = k / frac).")

#parser.add_argument("--erosion-iterations", default=0, type=int, help="Erosion # of iterations. Negative for dilation, zero for no erosion/dilation.")
#parser.add_argument("--erosion-kernel", default=5, type=int, help="Erosion/dilation kernel size.")

#parser.add_argument('--kmeans-epsilon', default=0.3, type=float, help="K-means clustering epsilon value. [1.0, 0.0]. Higher values seem to shrink the foreground.")
#parser.add_argument('--kmeans-iterations', default=100, type=int, help="K-means number of iterations.")
#parser.add_argument('--kmeans-inits', default=10, type=int, help="K-means # of times the algorithm is executed using different initial labelling.")

args = parser.parse_args()
BLACK = np.zeros((args.height, args.width, 3), dtype=np.uint8) # 2Dx3 uint8
BLACK_MONO = np.zeros((args.height, args.width), dtype=np.uint8) # 2Dx1 uint8
#assert(args.bg_blur_kernel == 0 or args.bg_blur_kernel%2 == 1)
#assert(1.0 >= args.kmeans_epsilon and args.kmeans_epsilon >= 0)
erosionKernel = lambda size: np.ones((size, size), np.uint8)

classifier = mediapipe.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) # model_selection info: https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294

def realSet(cam, prop, value):
    cam.set(prop, value)
    if value != int(cam.get(prop)):
        print(f'FATAL: prop {prop} failed to set to {value}!')
        quit()
camReal = cv2.VideoCapture(args.cam_path_real, cv2.CAP_V4L2)
#realSet(camReal, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(chr(0x59), chr(0x55), chr(0x59), chr(0x32))) # ffplay /dev/video0 says `Stream #0:0: Video: rawvideo (YUY2 / 0x32595559)`
#realSet(camReal, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*list('YUY2')))
#realSet(camReal, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*list('MJPG')))
realSet(camReal, cv2.CAP_PROP_FRAME_WIDTH, args.width)
realSet(camReal, cv2.CAP_PROP_FRAME_HEIGHT, args.height)
realSet(camReal, cv2.CAP_PROP_FPS, args.framerate)

# on demand mode
#inotify = INotify(nonblocking=True)
#wd = inotify.add_watch(args.cam_path_fake, flags.CREATE | flags.OPEN | flags.CLOSE_NOWRITE | flags.CLOSE_WRITE)
try:
    camFake = pyfakewebcam.FakeWebcam(args.cam_path_fake, args.width, args.height)
except FileNotFoundError as err:
    print(err)
    print("You probably just didn't run `sudo modprobe v4l2loopback` yet.") # `sudo modprobe v4l2loopback devices=2` for multiple devices

background = BLACK.copy()
background[:,:,1] = 255
master = tk.Tk()
sliderVals = {}
class Slider:
    def __init__(self, label, from_, to_, default, convert=int, step=None):
        if step:
            self.widget = tk.Scale(master, length=500, from_=from_, to=to_, orient=tk.HORIZONTAL, label=label, command=self.handle, resolution=step)
        else:
            self.widget = tk.Scale(master, length=500, from_=from_, to=to_, orient=tk.HORIZONTAL, label=label, command=self.handle)
        self.widget.pack()
        self.widget.set(default)
        sliderVals[label] = default
        self.label = label
        self.convert = convert
    def handle(self, val):
        global sliderVals
        val = self.convert(val)
        '''
        if self.label == 'blurKernel':
            val = int(val)
            if val != 0 and val%2 == 0:
                val += 1
            args.bg_blur_kernel = val
        elif self.label == 'erosionIterations':
            val = int(val)
            args.erosion_iterations = val
        elif self.label == 'kmeansEpsilon':
            val = float(val)
            args.kmeans_epsilon = val
        '''
        sliderVals[self.label] = val
'''
def tkScaleHandler(
def tksfBlurKernel(val):
    global args
    if val != 0 and val%2 == 0:
        val += 1
    args.bg_blur_kernel = val
def tksfErodeIterations(val):
    global args; args.erosion_iterations = val
def tksfKMeansEpsilon(val):
    global args; args.kmeans_epsilon = val

                self.widget = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, label='blurKern', command=tksfBlurKernel)
tksBlurKernel = 
tksErodeIterations = tk.Scale(master, from_=-20, to=20, orient=tk.HORIZONTAL, label='erodeIter', command=tksfErodeIterations)
tksKMeansEpsilon = tk.Scale(master, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, label='kmeansEps', command=tksfKMeansEpsilon)
for w in [tksBlurKernel, tksErodeIterations, tksKMeansEpsilon]:
    w.pack()
'''
def kernelSize(val):
    val = int(val)
    if val != 0 and val%2 == 0:
        return val + 1
    else:
        return val

for widgetParams in (
        ('blurKernel',        0, 100,   0, kernelSize),
        ('erodeIterations', -20,  20,   0),
        ('erodeKernel',       3,  25,   5, kernelSize),
        ('kmeansEpsilon',     0,   1, 0.3, float, 0.05),
        ('kmeansIterations',  1, 200, 100),
        ('kmeansInits',       1,  50,  10),
        ('blurSigma',         1, 100,   3)
):
    Slider(*widgetParams)

def applyMask(mask, fg, bg):
    assert(mask.shape == (args.height, args.width))
    assert(fg.shape == (args.height, args.width, 3))
    assert(bg.shape == (args.height, args.width, 3))
    maskBy3 = np.stack((mask,) * 3, axis=-1) # make a 2Dx1 into a 2Dx3
    frame = np.multiply(fg, maskBy3).astype(np.uint8)
    frame += np.multiply(bg, 1 - maskBy3).astype(np.uint8)
    return frame
    
def loop():
    try:
        # Read frame from real camera
        grabbed, frame = camReal.read()
        if not grabbed:
            master.after(500 / args.framerate, loop)
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 2Dx3 uint8
    
        # Run classifier
        frame.flags.writeable = False # pass by reference for better performance
        maskClassifier = classifier.process(frame).segmentation_mask # 2Dx1 float32
        frame.flags.writeable = True

        #if int(cv2.__version__[0]) > 3:
        contours, _ = cv2.findContours((maskClassifier * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #else:
        #    _, contours, _ = cv2.findContours((maskClassifier * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxContourAreaIndex = -1
        maxContourArea = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > maxContourArea:
                maxContourArea = area
                maxContourAreaIndex = i
        #code.interact(local=locals())
        # drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
        maskBlob = np.multiply(maskClassifier, cv2.drawContours(BLACK_MONO.copy(), contours, maxContourAreaIndex, (1.0), -1))
        
        # K-means cluster the mask into only 1 blob
        #kmeansCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, sliderVals['kmeansIterations'], sliderVals['kmeansEpsilon']) # finish on 100 iterations or less than 0.2 epsilon cluster movement
        # mask.reshape((-1, 1)) is 1D (width*height, 1)
        #_, labels, (centers) = cv2.kmeans(maskClassifier.reshape((-1, 1)), 2, None, kmeansCriteria, sliderVals['kmeansInits'], cv2.KMEANS_RANDOM_CENTERS) # mask, 1 cluster, None, stopping criteria, # of times the algorithm is executed using different initial labelling, centers
        #maskBlob = maskClassifier#centers[labels.flatten()].reshape((maskClassifier.shape))
    
        if sliderVals['erodeIterations'] > 0:
            mask = cv2.erode(maskBlob, erosionKernel(sliderVals['erodeKernel']), iterations=sliderVals['erodeIterations'])
        elif sliderVals['erodeIterations'] < 0:
            mask = cv2.dilate(maskBlob, erosionKernel(sliderVals['erodeKernel']), iterations=abs(sliderVals['erodeIterations']))
        else:
            mask = maskBlob
        if sliderVals['blurKernel'] != 0:
            mask = cv2.GaussianBlur(mask, (sliderVals['blurKernel'], sliderVals['blurKernel']), sliderVals['blurSigma'], borderType=cv2.BORDER_DEFAULT)
        #mask = cv2.blur(mask.astype(float), (50, 50))
    
        #frame = np.where(maskBy3 > 0.1, frame, background)
    
        #mask = cv2.threshold(mask, 0.75, 1, cv2.THRESH_BINARY).astype(np.uint8) # 1 = maxvalue
        #frame[:,:,0] = (mask * frame[:,:,0]).astype(np.uint8)
        #frame[:,:,1] = (mask * frame[:,:,1]).astype(np.uint8)
        #frame[:,:,2] = (mask * frame[:,:,2]).astype(np.uint8)
    
        #mask = (mask - mask.min()) / mask.ptp() # Optional. It never goes outside of 0-1, and it's only ever off by up to 3% away from 1 and 1% away from 0.
        #maskBy3 = np.stack((mask,) * 3, axis=-1) # make a 2Dx1 into a 2Dx3
        #print(mask.shape, maskBy3.shape)
        #frame = np.multiply(frame, maskBy3).astype(np.uint8)
        #frame += np.multiply(background, 1-maskBy3).astype(np.uint8)

        #frame = back.copy(background)
        #frame[:,:,1] = (mask * 255).astype(np.uint8)
        #frame[:,:,1] = (maskClassifier * 255).astype(np.uint8)
        #frame[:,:,2] = (maskKMeans * 255).astype(np.uint8)
        camFake.schedule_frame(applyMask(maskBlob, frame, BLACK))
        master.after(1, loop)
    except KeyboardInterrupt as err:
        print('Quitting.')
        master.destroy()
        camReal.release()
    except Exception as e:
        master.destroy()
        camReal.release()
        traceback.print_exc()
        
        
master.after(1, loop)
print('Main loop')
master.mainloop()
    
'''
blue = np.zeros((args.height, args.width, 3), dtype=np.uint8)
green = np.zeros((args.height, args.width, 3), dtype=np.uint8)
black = np.zeros((args.height, args.width, 3), dtype=np.uint8)
blue[:,:,2] = 255
green[:,:,1] = 255
frames = [blue, green, black]

code.interact(local=locals())
quit()

count = 0
while True:
    fake_cam.schedule_frame(frames[int(count / args.framerate) % 3])
    time.sleep(1 / args.framerate)
    count += 1


    
lastMask = None
mask = classifier.process(frame).segmentation_mask
mast = (mask > self.threshold) * mask

if self.postprocess:
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.blur(mask.astype(float), (10, 10))

if lastMask is None:
    lastMask = mask
mask = mask * 0.5 + self.old_mask * (1.0 - 0.5) # 0.5 = Mask Running Average Ratio
lastMask = mask

# Background blur
background_frame = cv2.GaussianBlur(frame, (args.bg_blur_kernel, args.bg_blur_kernel), args.bg_blur_sigma, borderType=cv2.BORDER_DEFAULT)
# Get background image
if self.no_background is False:
    background_frame = next(self.images["background"])
else:
'''
