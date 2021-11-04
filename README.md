Webcam background replacer. Other effects / features TBD, but background blurring / frosted glass effect planned.

Prereqs:
    sudo apt install v4l2loopback-utils
    
    numpy>=1.19.3
    opencv-python>=4.4.0.46
    pyfakewebcam>=0.1.0
    mediapipe>=0.8.7.1
    # optional: sudo pip3 install pyvirtualcam

How to run:
    # Assuming `ls /dev/video*` returns video0 and video1
    sudo modprobe v4l2loopback devices=1
    v4l2-ctl -d /dev/video2 -c sustain_framerate=1
    ./webcamBlur.py -b Downloads/bg_file.jpg &
    ffplay /dev/video2 # view the webcam

Make sure not to click the webcam view in Teams or it'll try to change to the nonexistent front-facing camera and lock-up Teams's webcam features.