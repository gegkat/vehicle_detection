import numpy as np
import cv2
import glob
import time

import matplotlib.pyplot as plt
from os.path import basename, splitext
from moviepy.editor import VideoFileClip, ImageSequenceClip
#
from feature_utils import *
from box_utils import *
from svc_utils import *
from VehicleDetector import *


class FeatureParams():
    def __init__(self):
        # colorspace
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

        # spatial features
        self.spatial_feat = True
        self.n_spatial = 16
        self.spatial_size = (self.n_spatial, self.n_spatial)

        # hist features
        self.hist_feat = True
        self.hist_bins = 1024

        # HOG tunables
        self.hog_feat = True
        self.orient = 9 # 9
        self.pix_per_cell = 8 # 8
        self.cell_per_block = 2 # 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

    def print(self):
        print('Using:', self.n_spatial, 'spatial', 
             self.hist_bins, 'hist_bins',
             self.orient,'orientations', 
             self.pix_per_cell, 'pixels per cell and', 
             self.cell_per_block,'cells per block')

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
    vehicle_detector = VehicleDetector(FeatureParams())

    fnames = glob.glob('test_images/*.jpg')
    # fnames = [fnames[2]]

    for fname in fnames:
        t1 = time.time()
        img = cv2.imread(fname)
        img_out, img_out2 = vehicle_detector.process_img(frame)
        t2 = time.time()
        print("{:.01f} seconds".format(t2-t1))
        plt.imshow(BGR2_(img_out, 'RGB'))
        plt.show()


def run(fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
    vehicle_detector = VehicleDetector(FeatureParams())

    clip = VideoFileClip(fname)
    n_frames = int(clip.fps * clip.duration)
    n_frames = min(MAX_FRAMES, n_frames) 
    count = 0
    images_list = []
    images_list2 = []

    for frame in clip.iter_frames():
        count = count+1
        if count % n_mod == 0:
            t1 = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            img_out, img_out2 = vehicle_detector.process_img(frame)
            img_out = BGR2_(img_out, 'RGB')
            images_list.append(img_out)

            img_out2 = cv2.addWeighted(img_out2, 1, img_out, 0.5, 0)
            images_list2.append(img_out2)


            t2 = time.time()
            print("{} of {} frames. {:.02f} seconds. {:.0f} of {} steps. {:.02f} seconds remain".format(
                count, n_frames, t2 - t1, count/n_mod, n_frames//n_mod, (n_frames//n_mod - count/n_mod)*(t2-t1)))
        if count >= MAX_FRAMES:
            break

    clip = ImageSequenceClip(images_list, fps=fps)


    savename = splitext(basename(fname))[0]
    savename = 'out_imgs/' + savename + '_out.mp4'
    clip.write_videofile(savename)

    clip2 = ImageSequenceClip(images_list2, fps=fps)
    savename = splitext(basename(fname))[0]
    savename = 'out_imgs/' + savename + '_out_heat.mp4'
    clip2.write_videofile(savename) 

    return clip