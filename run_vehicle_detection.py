import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from os.path import basename, splitext
from moviepy.editor import VideoFileClip, ImageSequenceClip
#
from feature_utils import *
from windows import *
from svc_utils import *
from Features import *

def train():
    # Divide up into cars and notcars
    vehicle_dirs = ['training_data/vehicles/GTI_MiddleClose',
                    'training_data/vehicles/GTI_Left',
                    'training_data/vehicles/GTI_Far',
                    'training_data/vehicles/GTI_Right',
                    'training_data/vehicles/KITTI_extracted'
                     ]

    non_vehicle_dirs = ['training_data/non-vehicles/GTI',
                        'training_data/non-vehicles/Extras',
                     ]
    cars = []
    notcars = []
    for currdir in vehicle_dirs:
        cars.extend(glob.glob(currdir + '/*.png'))

    for currdir in non_vehicle_dirs:
        notcars.extend(glob.glob(currdir + '/*.png'))

    # Shuffle data, only important if you are subsampling
    np.random.shuffle(cars)
    np.random.shuffle(notcars)

    # Reduce the sample size for test purposes
    sample_size = 20000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ## Tunable parameters

    # Load feature parameters
    features = Features()
    features.print()

    # Extract features
    t=time.time()
    car_features =    features.extract_features(cars)
    notcar_features = features.extract_features(notcars)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print('Feature vector length:', len(scaled_X[0]))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    train_svc(X_train, y_train, X_scaler)
    test_svc(X_test, y_test)


def test():
    svc, X_scaler = load_svc()
    img = cv2.imread('test_images/test6.jpg')

    windows = get_windows()
    print(len(windows))

    # img_out = draw_boxes(img, windows, color=(0, 0, 255), thick=6)
    # plt.imshow(BGR2_(img_out, 'RGB'))
    # plt.show()
    # return windows

    features = Features()

    t = time.time()
    on_windows = features.search_windows(img, windows, svc, X_scaler)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search windows...')

    img_out = draw_boxes(img, on_windows, color=(0, 0, 255), thick=6)
    plt.imshow(BGR2_(img_out, 'RGB'))
    plt.show()
    return on_windows, img_out


def run(fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
    svc, X_scaler = load_svc()

    windows = get_windows()
    features = Features()

    clip = VideoFileClip(fname)
    n_frames = int(clip.fps * clip.duration)
    n_frames = min(MAX_FRAMES, n_frames) 
    count = 0
    images_list = []
    for frame in clip.iter_frames():
        count = count+1
        if count % n_mod == 0:
            print("{} of {}".format(count, n_frames))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            on_windows = features.search_windows(frame, windows, svc, X_scaler)
            img_out = draw_boxes(frame, on_windows)
            img_out = BGR2_(img_out, 'RGB')
            images_list.append(img_out)
        if count >= MAX_FRAMES:
            break

    clip = ImageSequenceClip(images_list, fps=fps)

    savename = splitext(basename(fname))[0]
    savename = 'out_imgs/' + savename + '_out.mp4'
    clip.write_videofile(savename) 

    return clip
