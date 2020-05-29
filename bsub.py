#######################################################
#                                                     #
#                        BSUB                         #
#       A script to subtract static backgrounds       #
#######################################################
# Paula Florez Salcedo
# University of Utah 

# USAGE
# python3 bsub.py -v video_to_scan.mp4 
# python3 bsub.py

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threshhold",
                help="threshhold (optional) for background subtraction (default 15)")
ap.add_argument("-v", "--videopath", help="videopath is indicated", required=True)
ap.add_argument("-s","--show",help="Displays the current frame, will slow down the processing",action='store_true')
args = vars(ap.parse_args())


if not args.get("threshhold", False):
    thresh = 15

# otherwise, grab a reference to the video file
else:
    thresh = int(args.get("threshhold"))


cap = cv2.VideoCapture(args["videopath"])


frame_width = int(cap.get(3))
print(str(frame_width) + " is the width in pixels.")
frame_height = int(cap.get(4))
print(str(frame_height) + " is the height in pixels.")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args["videopath"][:-4]+"_blank"+args["videopath"][-4:], fourcc, 30, (frame_width, frame_height), False)

fps = cap.get(5)
frames = cap.get(7)
print(str(frames) + " is the total number of frames to be processed.")
cap.set(cv2.CAP_PROP_POS_FRAMES, frames - 100)
_, base_frame1 = cap.read()
first_gray = cv2.cvtColor(base_frame1, cv2.COLOR_RGB2GRAY)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

_, base_frame2 = cap.read()
second_gray = cv2.cvtColor(base_frame2, cv2.COLOR_RGB2GRAY)
cap.set(cv2.CAP_PROP_POS_FRAMES, frames*.375800227785293)

_, base_frame2 = cap.read()
third_gray = cv2.cvtColor(base_frame2, cv2.COLOR_RGB2GRAY)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

gray_diff = cv2.absdiff(first_gray,second_gray)
# cv2.imshow("1",gray_diff)
gray_diff = cv2.absdiff(third_gray,gray_diff)
# cv2.imshow("2",gray_diff)
print(cap.get(5))
framestate = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # difference = cv2.absdiff(first_gray, gray_frame)
        # difference = cv2.absdiff(base_frame1, frame)
        substract = cv2.subtract(gray_diff, gray_frame, mask=None)



###########
#
#    EDIT THE THRESHHOLD UNTIL THE BLANKING LOOKS GOOD
#
###########
        _, substract = cv2.threshold(substract, thresh, 255, cv2.THRESH_BINARY)

        # cv2.imshow("GraY DIFF", gray_diff)
        # cv2.imshow("First graye", first_gray)
        # cv2.imshow("second gray", second_gray)
        # cv2.imshow("thrid", third_gray)
        if(args.get("show")):
            cv2.imshow('Frame', gray_frame)
        # cv2.imshow('difference', difference)
            cv2.imshow("substract", substract)
        out.write(substract)

        framestate = framestate+1
        if(framestate%1000 == 0):
            print(str(framestate) + " of " + str(frames) + " frames processed.")
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    else:
        print("processing done")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
