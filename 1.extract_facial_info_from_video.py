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
args = ap.parse_args()


def variance_of_laplacian(image):
    """Return how blurry the input image is.

    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry_face(image, rect):
    """Return whether the input image is blurry or not."""
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0

    fm = variance_of_laplacian(gray[y:y+h, x:x+w])
    if fm < args.blurry_threshold:
        return True

    return False


def get_landmarks(shape):
    """Return landmarks for eyes, nose, and mouth."""
    p1x = (shape[36][0] + shape[39][0]) / 2
    p1y = (shape[36][1] + shape[39][1]) / 2
    p2x = (shape[42][0] + shape[45][0]) / 2
    p2y = (shape[42][1] + shape[45][1]) / 2
    (p3x, p3y) = shape[30]
    (p4x, p4y) = shape[48]
    (p5x, p5y) = shape[54]

    return [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y]


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    # columns and empty list to make a Pandas' dataframe
    cols, lst = ['NAME_ID', 'P1X', 'P1Y',
                            'P2X', 'P2Y',
                            'P3X', 'P3Y',
                            'P4X', 'P4Y',
                            'P5X', 'P5Y'], []

    # create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(args.video)
    success = True
    count = -1

    # loop over the frames from the input video
    while success:
        # process every args.interval image
        count += 1
        if count % args.interval != 0:
            continue

        print('preceeding frame_%d...' % (count))

        # grab the frame from the threaded video
        # and convert it to grayscale
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for i, rect in enumerate(rects):
            # focus measure of the image using the Variance of Laplacian method
            if is_blurry_face(gray, rect):
                continue

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a Numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # fetch (x, y)-coordinates for the facial landmarks
            landmarks = get_landmarks(shape)

            # save the frame
            save_path = args.output+"/frame_%d_%d.jpg" % (count, i)
            lst.append([save_path] + landmarks)
            cv2.imwrite(save_path, frame)

    # save landmarks as a csv file
    landmarks_info = pd.DataFrame(lst, columns=cols)
    landmarks_info.to_csv(args.output+'/landmarks.csv')

    # do a bit of cleanup
    cap.release()
