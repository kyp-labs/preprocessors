"""Video facial landmark detection with OpenCV, Python, and dlib.

Prediction Model: http://dlib.net/files/
                  shape_predictor_68_face_landmarks.dat.bz2
"""

# import the necessary packages
from imutils import face_utils
import argparse
import dlib
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-v', '--video', type=str, required=True,
                help='path to the video for facial detection')
args = ap.parse_args()


if __name__ == '__main__':
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print('[INFO] loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    # create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(args.video)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video
        # and convert it to grayscale
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convertt the facial landmark (x, y)-coordinates to a Numpy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()
