#######################################################
#                                                     #
#                   RUBY TRACKER                      #
#       A script to track objects in a video          #
#######################################################
# Robert Paul Baskin
# University of Utah 


# Adapted for research use from Adrian Rosebrock, OpenCV Object Track Object Movement, PyImageSearch,
# https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
# by Robert Paul Baskin at the Univeristy of Utah

# USAGE
# python ruby_tracker.py --video video_to_scan.mp4
# python ruby_tracker.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video",
                help="path to the video file", required=True)
parser.add_argument("-t", "--tail", type=int, default=64,
                help="tail length")
parser.add_argument("-s", "--show", help="Shows the frame being processed currently, slows down processing",
                action='store_true')
args = vars(parser.parse_args())

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["tail"])
out = []
counter = 0

# define the boundaries of the white color
sens = 15 
#whiteLow = np.array([0, 0, 255 - sens])
#whiteUp = np.array([255, sens, 255])

whiteLow = np.array([0, 0, 30])
whiteUp = np.array([360, 100, 100])


#grabs the video to begin processing
if not args.get("video", False):
    vs = VideoStream(src=0).start()
    outputfile = open("camera_video.csv", 'w')
else:
    vs = cv2.VideoCapture(args["video"])
    length = len(str(args.get("video", False)))
    outputfile = open('analyzed_' + str(args.get("video", False)[:length - 4] + '.csv'), 'w')

fps = vs.get(5)
totalframes = vs.get(7)
print(str(totalframes) + " total frames to be processed.")

framestate = 0
while True:
    # current frame
    frame = vs.read()

    # handle the frame from VideoCapture
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    # resize the frame, blur it, and convert it to HSV, this
    # allows us to input the color desired to be tracked (white)
    # in a range of colors.

    # essentially grabs a layer which contains all pixels within
    # the color range. Then scrubs the noise away
    
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (19, 19), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, whiteLow, whiteUp)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the shape
    counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    counts = imutils.grab_contours(counts)
    center = None

    # only proceed if at least one contour was found
    if len(counts) > 0:
        
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # this permits us to identify and follow the fish
        
        c = max(counts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        out.append(str(int(M["m10"] / M["m00"])) + ',' + str(int(M["m01"] / M["m00"])) + '\n')


        pts.appendleft(center)

    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["tail"] / float(i + 1)) * 2)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # shows the processing to visually verify that it is working. 
    if args.get("show"):
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    framestate = framestate + 1
    if (framestate % 1000 == 0):
        print(str(framestate) + " of " + str(totalframes) + " frames processed.")

    if key == ord("q"):
        break

if not args.get("video", False):
    i = 1
    for each in out:
        outputfile.write(str(i) + ',' + each[0:-1] + ',' + '=SQRT((B' + str(i + 1) + '-B' + str(i) + ')^2+(C' + str(
            i + 1) + '-C' + str(i) + ')^2)/' + str(fps) + '\n')
        i += 1
    vs.stop()

# write output
else:
    outputfile.write('Frame,Xpos,Ypos,Velocity,\n')
    i = 1
    for each in out:
        outputfile.write(str(i) + ',' + each[0:-1] + ',' + '=SQRT((B' + str(i + 2) + '-B' + str(i + 1) + ')^2+(C' + str(
            i + 2) + '-C' + str(i + 1) + ')^2)/' + str(fps) + '\n')
        i += 1
    print("processing finished successfully")
    vs.release()

cv2.destroyAllWindows()
