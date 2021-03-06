#!/usr/bin/env python3

import cv2, time, argparse, mediapipe, traceback, code
import numpy as np
import tkinter as tk

USE_NEW_CAM_LIB = False

parser = argparse.ArgumentParser(description='''Webcam background blurrer/replacer.
Run `sudo modprobe v4l2loopback devices=1 && v4l2-ctl -d /dev/video2 -c sustain_framerate=1` to setup the loopback device before you run this.
In Teams, DO NOT CLICK THE WEBCAM VIEW. That attempts to switch to front-facing camera, which seems to try to read the real webcam and crash because it's already in use.''')
parser.add_argument('-r', '--webcam-path-real', dest='cam_path_real', default="/dev/video0", help="Set real webcam path")
parser.add_argument('-f', '--webcam-path-fake', dest='cam_path_fake', default="/dev/video2", help="V4l2loopback device path")

parser.add_argument('-W', '--width', default=640, type=int, help="Resolution width.")
parser.add_argument('-H', '--height', default=480, type=int, help="Resolution height.")
parser.add_argument('-F', '--framerate', default=30, type=int, help="Framerate.")

parser.add_argument('-b', '--background-file', default='/home/tony/Downloads/spaceship_bg.jpg', help="Starting background file path. Can be changed dynamically in the GUI.")
parser.add_argument('-a', '--average-frames', default=5, type=int, help="Average the mask across this many frames.")

#parser.add_argument('--background-blur', dest='bg_blur_kernel', default=21, type=int, metavar='k', help="The gaussian bluring kernel size in pixels. MUST BE AN ODD NUMBER. Zero disables blurring.")
#parser.add_argument("--background-blur-sigma-frac", dest='bg_blur_sigma', default=3, type=int, metavar='frac', help="The fraction of the kernel size to use for the sigma value (ie. sigma = k / frac).")

#parser.add_argument("--erosion-iterations", default=0, type=int, help="Erosion # of iterations. Negative for dilation, zero for no erosion/dilation.")
#parser.add_argument("--erosion-kernel", default=5, type=int, help="Erosion/dilation kernel size.")

args = parser.parse_args()

classifier = mediapipe.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) # model_selection info: https://github.com/fangfufu/Linux-Fake-Background-Webcam/issues/135#issuecomment-883361294

def realSet(cam, prop, value):
    cam.set(prop, value)
    if value != int(cam.get(prop)):
        print(f'FATAL: prop {prop} failed to set to {value}! Got {cam.get(prop)}!')
        quit()
camReal = cv2.VideoCapture(args.cam_path_real, cv2.CAP_V4L2)
realSet(camReal, cv2.CAP_PROP_FPS, args.framerate)
realSet(camReal, cv2.CAP_PROP_FRAME_HEIGHT, args.height)
realSet(camReal, cv2.CAP_PROP_FRAME_WIDTH, args.width)

if USE_NEW_CAM_LIB:
    import pyvirtualcam
    camFake = pyvirtualcam.Camera(width=1280, height=720, fps=30)
    print(camFake.device)
else:
    import pyfakewebcam
    try:
        camFake = pyfakewebcam.FakeWebcam(args.cam_path_fake, args.width, args.height)
    except FileNotFoundError as err:
        print(err)
        print("You probably just didn't run `sudo modprobe v4l2loopback` yet.") # `sudo modprobe v4l2loopback devices=2` for multiple devices
# `sudo modprobe v4l2loopback devices=1 && v4l2-ctl -d /dev/video2 -c sustain_framerate=1`
# exclusive_caps=1 is incompatible with Teams
# https://github.com/umlaeute/v4l2loopback

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

master.title('WebcamEffects knobs')
bgFileInput = tk.Text(master, height=1, width=60)
bgFileInput.insert('1.0', args.background_file)
bgFileInput.pack()
background = None
def setBGFile():
    global bgFileInput, background
    try:
        fn = bgFileInput.get('1.0', 'end').strip()
        print(fn)
        img = cv2.imread(fn)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        background = cv2.resize(img, (args.width, args.height))
    except:
        traceback.print_exc()

setBGFile()
bgFileButton = tk.Button(master, text='Set BGFile', command=setBGFile)
bgFileButton.pack()

BLACK = np.zeros((args.height, args.width, 3), dtype=np.uint8) # 2Dx3 uint8
BLACK_MONO = np.zeros((args.height, args.width), dtype=np.uint8) # 2Dx1 uint8
#background = BLACK.copy()
#background[:,:,1] = 255


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

if USE_NEW_CAM_LIB:
    img = cv2.resize(background, (1280, 720))
    while True:
        camFake.send(img)
        time.sleep(1 / 100)
    quit()

erosionKernel = lambda size: np.ones((size, size), np.uint8)
maskAveraging = [BLACK_MONO] * args.average_frames
maskAveragingIndex = 0
def loop():
    global maskAveraging, maskAveragingIndex
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

        if args.average_frames > 0:
            maskAveraging[maskAveragingIndex] = mask
            maskAveragingIndex = (maskAveragingIndex + 1) % args.average_frames
            for i in range(args.average_frames - 1):
                mask += maskAveraging[(maskAveragingIndex + i) % args.average_frames]
            mask /= args.average_frames

        out = applyMask(mask, frame, background, sliderVals['maskThreshold'])
        if USE_NEW_CAM_LIB:
            camFake.send(cv2.resize(out, (1280, 720)))
        else:
            camFake.schedule_frame(out)
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
