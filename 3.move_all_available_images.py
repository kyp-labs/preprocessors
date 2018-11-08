"""Move all available images and generate a proper csv file."""


import os
import argparse
import shutil
import pandas as pd
import utils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image-pathes', type=str, nargs='+', required=True,
                help='pathes to the input with csv files')
ap.add_argument('-o', '--output', type=str, required=True,
                help='path to the output for face-centered images')
args = ap.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # columns and empty list to make a Pandas' dataframe
    cols, lst = utils.get_landmarks_dataframe_args()

    for image_path in args.image_pathes:
        print('[INFO] reading a csv file...')
        landmarks_info = pd.read_csv(image_path+'/landmarks.csv')

        for i in range(len(landmarks_info)):
            img_name = landmarks_info[i:i+1].values[0][1]

            if not os.path.exists(img_name):
                continue

            new_img_name = args.output + '/' + img_name.replace('/', '_')
            shutil.copy(img_name, new_img_name)
            landmarks = landmarks_info[i:i+1].values[0][2:].tolist()

            lst.append([new_img_name] + landmarks)

    # save landmarks as a csv feile
    print('[INFO] saving a csv file...')
    new_landmarks_info = pd.DataFrame(lst, columns=cols)
    new_landmarks_info.to_csv(args.output+'/landmarks.csv')
