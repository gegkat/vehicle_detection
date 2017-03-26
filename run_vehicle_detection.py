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
from scipy.ndimage.measurements import label
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


def pipeline(img, windows, svc, X_scaler, features):

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    #on_windows = features.search_windows(img, windows, svc, X_scaler)

    # img_out1, on_windows = features.find_cars(img, 400, 656, 2.0, svc, X_scaler)
    # heat = add_heat(heat,on_windows)

    # # img_out1, on_windows = features.find_cars(img, 390, 600, 1.4, svc, X_scaler)
    # # heat = add_heat(heat,on_windows)

    # img_out1, on_windows = features.find_cars(img, 390, 510, 0.8, svc, X_scaler)
    # heat = add_heat(heat,on_windows)


    #img_out1, on_windows = features.find_cars(img, 400, 656, 2.0, svc, X_scaler)
    #heat = add_heat(heat,on_windows)

    t1 = time.time()
    # # img_out1, on_windows = features.find_cars(img, 380, 650, 2.55, svc, X_scaler, (1,1))
    # # heat = add_heat(heat,on_windows)

    # img_out1, on_windows = features.find_cars(img, 380, 622, 2.15, svc, X_scaler, (2,1))
    # heat = add_heat(heat,on_windows)

    # img_out1, on_windows = features.find_cars(img, 410, 610, 1.8, svc, X_scaler, (1,1))
    # heat = add_heat(heat,on_windows)

    # # img_out1, on_windows = features.find_cars(img, 410, 580, 1.4, svc, X_scaler, (1,1))
    # # heat = add_heat(heat,on_windows)

    # # img_out1, on_windows = features.find_cars(img, 390, 510, .8, svc, X_scaler)
    # # heat = add_heat(heat,on_windows)

    # img_out1, on_windows = features.find_cars(img, 390, 550, 1.2, svc, X_scaler, (3,3))
    # heat = add_heat(heat,on_windows)


    img_out1, on_windows = features.find_cars(img, 380, 622, 1.8, svc, X_scaler, (1,1))
    heat = add_heat(heat,on_windows)

    img_out1, on_windows = features.find_cars(img, 410, 610, 1.5, svc, X_scaler, (2,1))
    heat = add_heat(heat,on_windows)

    # img_out1, on_windows = features.find_cars(img, 390, 550, 1.5, svc, X_scaler, (2,2))
    # heat = add_heat(heat,on_windows)


    t2 = time.time()
    # print(t2-t1)
    # plt.imshow(heat)

    return heat

def heat_to_img(img, heat, threshold):
    heat_thresh = apply_threshold(heat, threshold)
    labels = label(heat_thresh)
    img_out = draw_labeled_bboxes(img, labels)
    return img_out, heat_thresh

def test():
    svc, X_scaler = load_svc()
    features = Features()

    fnames = glob.glob('test_images/*.jpg')
    # fnames = [fnames[2]]
    windows = get_windows()
    windows = []
    print(len(windows))

    for fname in fnames:
        t1 = time.time()
        img = cv2.imread(fname)
        heat = pipeline(img, windows, svc, X_scaler, features)
        img_out = heat_to_img(img, heat)
        t2 = time.time()
        print("{:.01f} seconds".format(t2-t1))
        plt.imshow(BGR2_(img_out, 'RGB'))
        plt.show()

def test2():
    svc, X_scaler = load_svc()
    img = cv2.imread('test_images/test6.jpg')

    features = Features()

    t = time.time()

    # windows.extend(slide_window(img_shape, y_start_stop=[400, 700], xy_window=(sz*3, sz*3), xy_overlap=overlap))
    # windows.extend(slide_window(img_shape, y_start_stop=[400, 550], xy_window=(sz*2, sz*2), xy_overlap=overlap))
    # windows.extend(slide_window(img_shape, y_start_stop=[400, 500], xy_window=(sz*1, sz*1), xy_overlap=overlap))

    img_out = features.find_cars(img,     720, 400, 0.5, svc, X_scaler)
 #   img_out = features.find_cars(img, 400, 550, 0.5, svc, X_scaler)
 #   img_out = features.find_cars(img, 400, 500, 1.0, svc, X_scaler)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to search windows...')

    # img_out = draw_boxes(img, on_windows, color=(0, 0, 255), thick=6)
    plt.imshow(BGR2_(img_out, 'RGB'))
    plt.show()
    return img_out


def run(fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
    svc, X_scaler = load_svc()

    windows = get_windows()
    print(len(windows))
    features = Features()

    clip = VideoFileClip(fname)
    n_frames = int(clip.fps * clip.duration)
    n_frames = min(MAX_FRAMES, n_frames) 
    count = 0
    images_list = []
    images_list2 = []

    MAP_MAX = 10
    MAP_MIN = -30
    global_heat = None
    for frame in clip.iter_frames():


        count = count+1
        if count % n_mod == 0:

            t1 = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            if global_heat is None:
                global_heat = np.zeros_like(frame[:,:,0]).astype(np.float)
            curr_heat = pipeline(frame, windows, svc, X_scaler, features)
            # print(np.max(curr_heat))
            global_heat += curr_heat-2
            # global_heat = curr_heat

            global_heat[global_heat < MAP_MIN] = MAP_MIN
            global_heat[global_heat > MAP_MAX] = MAP_MAX
            img_out = frame
            img_out, heat_thresh = heat_to_img(frame, global_heat, 0)

            img_out = BGR2_(img_out, 'RGB')
            images_list.append(img_out)

            tmp = np.copy(global_heat)
            tmp = (tmp - MAP_MIN)*255/(MAP_MAX - MAP_MIN)
            tmp *= 0.8

            img_out2 = np.zeros_like(img_out)
            img_out2[:,:,0] = tmp
            img_out2[:,:,1] = heat_thresh*255/np.max(heat_thresh)
            # img_out2[:,:,2] = tmp

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
