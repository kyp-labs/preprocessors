"""Face-centered image generator with facial landmark information.

The implementaion is a little modified from:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
"""

import os
import PIL
import argparse
import numpy as np
import pandas as pd
from scipy import ndimage
import utils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image-path', required=True,
                help='path to source images')
ap.add_argument('-o', '--output', type=str, required=True,
                help='path to the output for face-centered images')
ap.add_argument('-r', '--resolution', type=int, default=128,
                help='output resolution')
args = ap.parse_args()


def rot90(v):
    """Rotate the input vector 90 degrees."""
    return np.array([-v[1], v[0]])


def resize_landmarks(landmarks, quad):
    """Resize the input landmarks."""
    size = ((quad[3][0] + quad[2][0]) - (quad[0][0] + quad[1][0])) / 2
    return landmarks * args.resolution / size


def rotate_landmarks(landmarks, quad):
    """Rotate the inputt landmarks."""
    hypot_vec = quad[1] - quad[0]
    bottom_vec = np.array([0., hypot_vec[1]])
    height_vec = hypot_vec - bottom_vec
    sin = np.hypot(*height_vec) / np.hypot(*hypot_vec)
    cos = np.hypot(*bottom_vec) / np.hypot(*hypot_vec)

    # Rotate clockwise
    if height_vec[0] > 0:
        sin *= -1.

    rotation_matrix = np.matrix([[cos, -sin], [sin, cos]])
    return landmarks.dot(rotation_matrix)


def generate_face_centered_images(img, loose_landmarks):
    """Generate face-centered image with newly caculated landmarks."""
    landmarks = np.stack([loose_landmarks[2*i:2*i+2]
                         for i in range(68)]).astype('float32')

    landmarks_for_centered_face =\
        utils.get_landmarks_for_centered_face(landmarks)
    left_eye = landmarks_for_centered_face[0]
    right_eye = landmarks_for_centered_face[1]
    left_mouth = landmarks_for_centered_face[3]
    right_mouth = landmarks_for_centered_face[4]

    # Choose oriented crop rectangle.
    eye_avg = (left_eye + right_eye) * 0.5 + 0.5
    mouth_avg = (left_mouth + right_mouth) * 0.5 + 0.5
    eye_to_eye = right_mouth - left_mouth
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = args.resolution / (np.hypot(*x) * 2)

    # Shrink.
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        size = (int(np.round(float(img.size[0]) / shrink)),
                int(np.round(float(img.size[1]) / shrink)))
        img = img.resize(size, PIL.Image.ANTIALIAS)
        quad /= shrink
        landmarks /= shrink
        zoom *= shrink

    # Crop.
    border = max(int(np.round(args.resolution * 0.1 / zoom)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
        landmarks -= crop[0:2]

    # Simulate super-resolution.
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize((img.size[0] * superres, img.size[1] * superres),
                         PIL.Image.ANTIALIAS)
        quad *= superres
        landmarks *= superres
        zoom /= superres

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(args.resolution * 0.3 / zoom)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]),
                     (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0],
                                np.float32(y) / pad[1]),
                                np.minimum(np.float32(w-1-x) / pad[2],
                                np.float32(h-1-y) / pad[3]))
        blur = args.resolution * 0.02 / zoom
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) *\
            np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) *\
            np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)),
                                  'RGB')
        quad += pad[0:2]
        landmarks += pad[0:2]

    # Transform.
    quad += 0.5
    img = img.transform((args.resolution*4, args.resolution*4),
                        PIL.Image.QUAD,
                        quad.flatten(), PIL.Image.BILINEAR)
    img = img.resize((args.resolution, args.resolution),
                     PIL.Image.ANTIALIAS)

    landmarks -= quad[0]
    landmarks = rotate_landmarks(landmarks, quad)
    landmarks = resize_landmarks(landmarks, quad)

    # Save new landmarks
    new_landmarks = landmarks.reshape(-1).tolist()[0]
    print(new_landmarks)

    return (img, new_landmarks)


if __name__ == '__main__':
    assert(os.path.exists(args.image_path))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    landmarks_info = pd.read_csv(args.image_path+'/landmarks.csv')

    # columns and empty list to make a Pandas' dataframe
    cols, lst = utils.get_landmarks_dataframe_args()

    for i, img_name in enumerate(landmarks_info['NAME_ID']):
        print('preceeding %s...' % (img_name))

        img = PIL.Image.open(img_name)
        loose_landmarks =\
            landmarks_info[i:i+1].values[0][2:138].astype('float32')

        centered_img, new_landmarks =\
            generate_face_centered_images(img, loose_landmarks)

        output_img_name = args.output + '/' + img_name.split('/')[1]
        centered_img.save(output_img_name)
        lst.append([output_img_name] + new_landmarks)

    # save landmarks as a csv file
    new_landmarks_info = pd.DataFrame(lst, columns=cols)
    new_landmarks_info.to_csv(args.output+'/landmarks.csv')
