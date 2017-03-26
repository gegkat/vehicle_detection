import numpy as np
import cv2
import glob
import time

import matplotlib.pyplot as plt
from os.path import basename, splitext
from moviepy.editor import VideoFileClip, ImageSequenceClip
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

#
import VehicleDetector 
from feature_utils import *
from svc_utils import *

# Class to hold constants defining features for SVC training
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
    feature_params = FeatureParams()
    feature_params.print()

    # Extract features
    t=time.time()
    car_features =    extract_features(feature_params, cars)
    notcar_features = extract_features(feature_params, notcars)
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

# Test the pipeline on still iamges
def test(fname='test_images/*.jpg'):

    # Get list of images
    fnames = glob.glob(fname)
    for fname in fnames:
        # Initialize new VehicleDetector object
        vehicle_detector = VehicleDetector.VehicleDetector(FeatureParams())

        # Read test image
        img = cv2.imread(fname)

        # process_img
        t1 = time.time()
        vehicle_detector.process_img(img)
        t2 = time.time()
        print("{:.02f} seconds".format(t2-t1))

        # Make plots
        vehicle_detector.plot_images()
        vehicle_detector.plot()

# Function to process a video stream
def run(do_plot=False, fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):

    # Initialize VehicleDetector object
    vehicle_detector = VehicleDetector.VehicleDetector(FeatureParams())

    # Open video file
    clip = VideoFileClip(fname)

    # Calculate video # of frames
    n_frames = int(clip.fps * clip.duration)
    n_frames = min(MAX_FRAMES, n_frames) 

    # Initialize frame counter
    count = 0

    # Initialize output video image lists
    images_list = []
    images_list2 = []

    # Loop through all frames
    for frame in clip.iter_frames():
        # increment counter
        count = count+1

        # Process only every n_mod frame
        if count % n_mod == 0:

            t1 = time.time()

            # Get frame and convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

            # Process image
            vehicle_detector.process_img(frame)

            # Append img_out for output video
            images_list.append(vehicle_detector.img_out)

            # Plot if desired
            if do_plot:
                vehicle_detector.plot_images()
                images_list2.append(vehicle_detector.draw_img_found)

            t2 = time.time()
            print("{} of {} frames. {:.02f} seconds. {:.0f} of {} steps. {:.02f} seconds remain".format(
                count, n_frames, t2 - t1, count/n_mod, n_frames//n_mod, (n_frames//n_mod - count/n_mod)*(t2-t1)))

        # Break loop if MAX_FRAMES reached
        if count >= MAX_FRAMES:
            break

    # Save output video
    clip = ImageSequenceClip(images_list, fps=fps)
    savename = splitext(basename(fname))[0]
    savename = 'out_imgs/' + savename + '_out.mp4'
    clip.write_videofile(savename)

    # Optionally save extra output video
    if do_plot:
        clip2 = ImageSequenceClip(images_list2, fps=fps)
        savename = splitext(basename(fname))[0]
        savename = 'out_imgs/' + savename + '_out_plot.mp4'
        clip2.write_videofile(savename) 