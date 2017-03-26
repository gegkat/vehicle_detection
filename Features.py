import numpy as np
import cv2
from skimage.feature import hog
import time
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
        self.pix_per_cell = 8 # 8
        self.cell_per_block = 2 # 2
        self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

    def print(self):
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
        feature_image = BGR2_(img, self.color_space)

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

    # Define a function you will pass an image 
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, clf, scaler):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))     

            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)

            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))

            #6) Predict using your classifier
            prediction = clf.predict(test_features)

            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, skips):

        draw_img = np.copy(img)
        #img = img.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = BGR2_(img_tosearch, self.color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.pix_per_cell)-1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        # print(nxsteps, nysteps)
        
        t1 = time.time()
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        t2 = time.time()
        # print("{} seconds".format(t2-t1))  
        on_windows = []
        for xb in range(nxsteps):
            if xb % skips[0] == 0:
                for yb in range(nysteps):
                    if yb % skips[1] == 0:
                        ypos = yb*cells_per_step
                        xpos = xb*cells_per_step
                        # Extract HOG for this patch
                        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                        xleft = xpos*self.pix_per_cell
                        ytop = ypos*self.pix_per_cell

                        # Extract the image patch
                        subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                      
                        # Get color features
                        spatial_features = bin_spatial(subimg, size=self.spatial_size)
                        hist_features = color_hist(subimg, nbins=self.hist_bins)

                        # Scale features and make a prediction
                        test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                        test_prediction = svc.predict(test_features)
                        # test_prediction = 1
                        
                        if test_prediction == 1:
                            xbox_left = np.int(xleft*scale)
                            ytop_draw = np.int(ytop*scale)
                            win_draw = np.int(window*scale)
                            rect = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                            cv2.rectangle(draw_img,rect[0], rect[1],(0,0,255),6) 
                            on_windows.append(rect)
                        else:
                            xbox_left = np.int(xleft*scale)
                            ytop_draw = np.int(ytop*scale)
                            win_draw = np.int(window*scale)
                            rect = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                            cv2.rectangle(draw_img,rect[0], rect[1],(255,0,0),6) 
          

        return draw_img, on_windows
        
