# Video preprocessor for GAN implementation
Video preprocessor implementation for KYP team's projects.

## How to use

1. Extract facial landmarks and corresponding frames from an input video.
```bash
$ python 1.extract_facial_info_from_video.py -p models/shape_predictor_68_face_landmarks.dat -v INPUT_VIDEO_PATH -o OUTPUT_PATH
```

2. Generate facial centered images and an adjusted csv file (default size: 128 x 128)
```bash
$ python 2.generate_face_centered_images.py -i INPUT_FRAME_PATH_WITH_LANDMARK_CSV -o OUTPUT_PATH
```

3. Human-eye inspection: Remove low quality images from the output directory from the 2nd stage.

4. Integrate csv files and all output images from the stage 3.
```bash
$ python 3.move_all_available_images.py -i [INPUT_DIRECTORY_PATHES] -o OUTPUT_PATH
```
