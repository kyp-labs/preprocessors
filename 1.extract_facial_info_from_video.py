"""Video facial landmark detection with OpenCV, Python, and dlib.

Prediction Model: http://dlib.net/files/
                  shape_predictor_68_face_landmarks.dat.bz2
"""

# import the necessary packages
from imutils import face_utils
import pandas as pd
import argparse
import dlib
import cv2
import os
import utils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-v', '--video', type=str, required=True,
                help='path to the video for facial detection')
ap.add_argument('-o', '--output', type=str, required=True,
                help='path to the output for facial information')
ap.add_argument('-i', '--interval', type=int, default=20,
                help='frame interval not to store too much images')
ap.add_argument('-b', '--blurry-threshold', type=float, default=100.0,
                help='measures below this value will be considered blurry')
ap.add_argument('-m', '--max-output-num', type=int, default=-1,
                help='maximum number of output images')
args = ap.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    # columns and empty list to make a Pandas' dataframe
    cols, lst = utils.get_landmarks_dataframe_args()

    # create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(args.video)
    success = True
    count = -1
    out_count = 1

    # loop over the frames from the input video
    while success:
        # process every args.interval image
        count += 1
        if count % args.interval != 0:
            continue

        print('[INFO] preceeding frame_%d...' % (count))

        # grab the frame from the threaded video
        # and convert it to grayscale
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for i, rect in enumerate(rects):
            # focus measure of the image using the Variance of Laplacian method
            if utils.is_blurry_face(gray, rect, args.blurry_threshold):
                continue

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a Numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # fetch (x, y)-coordinates for the facial landmarks
            landmarks = [p for coordinates in shape for p in coordinates]

            # save the frame
            save_path = args.output+"/frame_%d_%d.jpg" % (count, i)
            lst.append([save_path] + landmarks)
            cv2.imwrite(save_path, frame)
            out_count += 1

        if args.max_output_num >= 0 and\
                args.max_output_num == out_count:
                    break

    # save landmarks as a csv file
    landmarks_info = pd.DataFrame(lst, columns=cols)
    landmarks_info.to_csv(args.output+'/landmarks.csv')

    # do a bit of cleanup
    cap.release()
