import numpy as np
import cv2
from scipy.ndimage.measurements import label
from feature_utils import *

# Increment heat map inside all boxes
def add_heat(heatmap, bbox_list):

    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
# Return a thresholded heat map
def apply_threshold(heatmap, threshold):

    # make a copy of heatmap
    heatmap_thresh = np.copy(heatmap)

    # Zero out pixels below the threshold
    heatmap_thresh[heatmap_thresh <= threshold] = 0

    # Return thresholded map
    return heatmap_thresh

# Draw boxes around labels in img
def draw_labeled_bboxes(img, labels):

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, feature_params, window_params, svc, X_scaler):

    # Unpack some of window_params
    ystart = window_params.ylims[0]
    ystop  = window_params.ylims[1]
    scale = window_params.scale
    
    xstart = 0
    xstop = 1280

    # limit img to between ystart and ystop
    img_tosearch = img[ystart:ystop, xstart:xstop,:]

    # Apply color conversion
    ctrans_tosearch = BGR2_(img_tosearch, feature_params.color_space)

    # Adjust for scale factor
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Break img into channels
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // feature_params.pix_per_cell)-1
    nyblocks = (ch1.shape[0] // feature_params.pix_per_cell)-1 
    nfeat_per_block = feature_params.orient*feature_params.cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // feature_params.pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, feature_params.orient, feature_params.pix_per_cell, feature_params.cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, feature_params.orient, feature_params.pix_per_cell, feature_params.cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, feature_params.orient, feature_params.pix_per_cell, feature_params.cell_per_block, feature_vec=False)

    # loop through all x and y steps and test each window
    on_windows = []
    all_windows = []
    for xb in range(0, nxsteps, window_params.xy_skips[0]):
        for yb in range(0, nysteps, window_params.xy_skips[1]):
            # Get current star pos
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            # Recombine hog from each channel
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Resample at pix_per_cell
            xleft = xpos*feature_params.pix_per_cell
            ytop = ypos*feature_params.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=feature_params.spatial_size)
            hist_features = color_hist(subimg, nbins=feature_params.hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

            # Use svc to predict
            test_prediction = svc.predict(test_features)
            
            # Calculate rectangles
            xbox_left = np.int(xleft*scale + xstart)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            rect = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))

            # Keep track of all windows that were searched
            all_windows.append(rect)

            # Keep track of windows returned with a positive prediction
            if test_prediction == 1:
                on_windows.append(rect)

    # Return draw images
    return on_windows, all_windows
    
