import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from feature_utils import *

class Features():
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
        self.pix_per_cell = 32 # 8
        self.cell_per_block = 2 # 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

        print('Using:', self.n_spatial, 'spatial', 
             self.hist_bins, 'hist_bins',
             self.orient,'orientations', 
             self.pix_per_cell, 'pixels per cell and', 
             self.cell_per_block,'cells per block')

    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img):    

        #1) Define an empty list to receive features
        img_features = []

        #2) Apply color conversion if other than 'RGB'
        feature_image = RGB2_(img, self.color_space)

        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)

            #4) Append features to list
            img_features.append(spatial_features)

        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)

            #6) Append features to list
            img_features.append(hist_features)

        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        self.orient, self.pix_per_cell, self.cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], self.orient, 
                            self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)

            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(file)
            img_features = self.single_img_features(image)
            features.append(img_features)
        # Return list of feature vectors
        return features
    
# # Define a function you will pass an image 
# # and the list of windows to be searched (output of slide_windows())
# def search_windows(img, windows, clf, scaler, color_space='RGB', 
#                     spatial_size=(32, 32), hist_bins=32, 
#                     hist_range=(0, 256), orient=9, 
#                     pix_per_cell=8, cell_per_block=2, 
#                     hog_channel=0, spatial_feat=True, 
#                     hist_feat=True, hog_feat=True):

#     #1) Create an empty list to receive positive detection windows
#     on_windows = []
#     #2) Iterate over all windows in the list
#     for window in windows:
#         #3) Extract the test window from original image
#         test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
#         #4) Extract features for that window using single_img_features()
#         features = single_img_features(test_img, color_space=color_space, 
#                             spatial_size=spatial_size, hist_bins=hist_bins, 
#                             orient=orient, pix_per_cell=pix_per_cell, 
#                             cell_per_block=cell_per_block, 
#                             hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                             hist_feat=hist_feat, hog_feat=hog_feat)
#         #5) Scale extracted features to be fed to classifier
#         test_features = scaler.transform(np.array(features).reshape(1, -1))
#         #6) Predict using your classifier
#         prediction = clf.predict(test_features)
#         #7) If positive (prediction == 1) then save the window
#         if prediction == 1:
#             on_windows.append(window)
#     #8) Return windows for positive detections
#     return on_windows

def train_hog_features():
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

    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 2000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ## Tunable parameters

    # Load feature parameters
    features = Features()

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

    train_svc(X_train, y_train)
    test_svc(X_test, y_test)

def train_svc(X_train, y_train):
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    t=time.time()
    svc_pickle = {}
    svc_pickle["svc"] = svc
    pickle.dump( svc_pickle, open( "svc.p", "wb" ) )
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to save SVC to pickle...')

    return svc

def test_svc(X_test, y_test):
    t=time.time()
    svc_pickle = pickle.load( open( "svc.p", "rb" ) )
    svc = svc_pickle["svc"]
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to load SVC from pickle...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# def test_svc(X_test, y_test):
#     svc_pickle = pickle.load( open( "svc.p", "rb" ) )
#     svc = svc_pickle["svc"]
