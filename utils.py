"""Helper functions needed for preprocessing."""


from imutils import face_utils
import numpy as np
import cv2


def get_landmarks_dataframe_args():
    """Return columns and empty list to make a Pandas' dataframe."""
    # detailed info of the 68 landmarks:
    # https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    cols, lst = ['NAME_ID', 'P1X', 'P1Y', 'P2X', 'P2Y',
                            'P3X', 'P3Y', 'P4X', 'P4Y',
                            'P5X', 'P5Y', 'P6X', 'P6Y',
                            'P7X', 'P7Y', 'P8X', 'P8Y',
                            'P9X', 'P9Y', 'P10X', 'P10Y',
                            'P11X', 'P11Y', 'P12X', 'P12Y',
                            'P13X', 'P13Y', 'P14X', 'P14Y',
                            'P15X', 'P15Y', 'P16X', 'P16Y',
                            'P17X', 'P17Y', 'P18X', 'P18Y',
                            'P19X', 'P19Y', 'P20X', 'P20Y',
                            'P21X', 'P21Y', 'P22X', 'P22Y',
                            'P23X', 'P23Y', 'P24X', 'P24Y',
                            'P25X', 'P25Y', 'P26X', 'P26Y',
                            'P27X', 'P27Y', 'P28X', 'P28Y',
                            'P29X', 'P29Y', 'P30X', 'P30Y',
                            'P31X', 'P31Y', 'P32X', 'P32Y',
                            'P33X', 'P33Y', 'P34X', 'P34Y',
                            'P35X', 'P35Y', 'P36X', 'P36Y',
                            'P37X', 'P37Y', 'P38X', 'P38Y',
                            'P39X', 'P39Y', 'P40X', 'P40Y',
                            'P41X', 'P41Y', 'P42X', 'P42Y',
                            'P43X', 'P43Y', 'P44X', 'P44Y',
                            'P45X', 'P45Y', 'P46X', 'P46Y',
                            'P47X', 'P47Y', 'P48X', 'P48Y',
                            'P49X', 'P49Y', 'P50X', 'P50Y',
                            'P51X', 'P51Y', 'P52X', 'P52Y',
                            'P53X', 'P53Y', 'P54X', 'P54Y',
                            'P55X', 'P55Y', 'P56X', 'P56Y',
                            'P57X', 'P57Y', 'P58X', 'P58Y',
                            'P59X', 'P59Y', 'P60X', 'P60Y',
                            'P61X', 'P61Y', 'P62X', 'P62Y',
                            'P63X', 'P63Y', 'P64X', 'P64Y',
                            'P65X', 'P65Y', 'P66X', 'P66Y',
                            'P67X', 'P67Y', 'P68X', 'P68Y'], []
    return (cols, lst)


def get_landmarks_for_centered_face(landmarks):
    """Return landmarks for eyes, nose, and mouth."""
    left_eye = (landmarks[36] + landmarks[39]) / 2
    right_eye = (landmarks[42] + landmarks[45]) / 2
    nose = landmarks[30]
    mouth_left = landmarks[48]
    mouth_right = landmarks[54]

    return np.stack([left_eye, right_eye, nose,
                     mouth_left, mouth_right])


def variance_of_laplacian(image):
    """Return how blurry the input image is.

    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry_face(image, rect, blurry_threshold):
    """Return whether the input image is blurry or not."""
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0

    fm = variance_of_laplacian(image[y:y+h, x:x+w])
    if fm < blurry_threshold:
        return True

    return False
