#!/usr/bin/env python3

import cv2, pyfakewebcam, time, argparse, mediapipe, traceback, code
import numpy as np
import tkinter as tk

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

args = parser.parse_args()

classifier = mediapipe.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) # model_selection info: https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294

def realSet(cam, prop, value):
    cam.set(prop, value)
    if value != int(cam.get(prop)):
        print(f'FATAL: prop {prop} failed to set to {value}!')
        quit()
camReal = cv2.VideoCapture(args.cam_path_real, cv2.CAP_V4L2)
realSet(camReal, cv2.CAP_PROP_FRAME_WIDTH, args.width)
realSet(camReal, cv2.CAP_PROP_FRAME_HEIGHT, args.height)
realSet(camReal, cv2.CAP_PROP_FPS, args.framerate)

try:
    camFake = pyfakewebcam.FakeWebcam(args.cam_path_fake, args.width, args.height)
except FileNotFoundError as err:
    print(err)
    print("You probably just didn't run `sudo modprobe v4l2loopback` yet.") # `sudo modprobe v4l2loopback devices=2` for multiple devices


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
        sliderVals[self.label] = val

def kernelSize(val):
    val = int(val)
    if val != 0 and val%2 == 0:
        return val + 1
    else:
        return val

for widgetParams in (
        ('classifierThreshold', 0.0, 1.0, 0.8, float, 0.01),
        ('erodeIterations', -20,  20,   0),
        ('erodeKernel',       3,  25,   5, kernelSize),
        ('blurKernel',        0, 100,  31, kernelSize),
        ('blurSigma',         1, 60,   5),
        ('maskThreshold',   0.0, 0.5, 0.0, float, 0.05),
):
    Slider(*widgetParams)



BLACK = np.zeros((args.height, args.width, 3), dtype=np.uint8) # 2Dx3 uint8
BLACK_MONO = np.zeros((args.height, args.width), dtype=np.uint8) # 2Dx1 uint8
#background = BLACK.copy()
#background[:,:,1] = 255
background = cv2.resize(cv2.imread('/home/tony/Downloads/DarkForest.jpg'), (args.width, args.height))


def applyMask(mask, fg, bg, threshold):
    assert(mask.shape == (args.height, args.width))
    assert(fg.shape == (args.height, args.width, 3))
    assert(bg.shape == (args.height, args.width, 3))
    if mask.min() > 0.02 or 0.0 > mask.min() or mask.ptp() > 1.001:
        print('Bad mask range:', mask.min(), mask.min() + mask.ptp())
        mask = (mask - mask.min()) / mask.ptp() # Optional. It never goes outside of 0-1, and it's usually very close to 0-1.
    maskBy3 = np.stack((mask,) * 3, axis=-1) # make a 2Dx1 into a 2Dx3
    #assert(maskBy3.shape == (args.height, args.width, 3))
    if threshold == 0.0:
        #return (maskBy3 * 255).astype(np.uint8)
        return (np.multiply(fg, maskBy3) + np.multiply(bg, 1 - maskBy3)).astype(np.uint8)
    else:
        return np.where(maskBy3 > threshold, fg, bg)

erosionKernel = lambda size: np.ones((size, size), np.uint8)
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

        _, maskBlob = cv2.threshold(maskClassifier, sliderVals['classifierThreshold'], 1, cv2.THRESH_BINARY) # 1 = maxvalue

        if sliderVals['erodeIterations'] > 0:
            mask = cv2.erode(maskBlob, erosionKernel(sliderVals['erodeKernel']), iterations=sliderVals['erodeIterations'])
        elif sliderVals['erodeIterations'] < 0:
            mask = cv2.dilate(maskBlob, erosionKernel(sliderVals['erodeKernel']), iterations=abs(sliderVals['erodeIterations']))
        else:
            mask = maskBlob
        if sliderVals['blurKernel'] != 0:
            mask = cv2.GaussianBlur(mask, (sliderVals['blurKernel'], sliderVals['blurKernel']), sliderVals['blurSigma'], borderType=cv2.BORDER_DEFAULT)



        #maskBy3 = np.stack((mask,) * 3, axis=-1) # make a 2Dx1 into a 2Dx3
        #print(mask.shape, maskBy3.shape)
        #frame = np.multiply(frame, maskBy3).astype(np.uint8)
        #frame += np.multiply(background, 1-maskBy3).astype(np.uint8)

        #frame = back.copy(background)
        #frame[:,:,1] = (mask * 255).astype(np.uint8)
        #frame[:,:,1] = (maskClassifier * 255).astype(np.uint8)
        #frame[:,:,2] = (maskKMeans * 255).astype(np.uint8)
        camFake.schedule_frame(applyMask(mask, frame, background, sliderVals['maskThreshold']))
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
