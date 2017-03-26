import numpy as np
import cv2
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    return cv2.resize(img, size).ravel() 

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

def BGR2_(img, color_space):
    color_options = {
        'BGR':   0,
        'RGB':   cv2.COLOR_BGR2RGB,
        'HSV':   cv2.COLOR_BGR2HSV,
        'LUV':   cv2.COLOR_BGR2LUV,
        'HLS':   cv2.COLOR_BGR2HLS,
        'YUV':   cv2.COLOR_BGR2YUV,
        'YCrCb': cv2.COLOR_BGR2YCrCb
    }

    conversion = color_options.get(color_space)
    if conversion == None:
        raise Exception('Did not recognize color space')
    elif conversion is 0:
        return np.copy(img) 
    else:
        return cv2.cvtColor(img, conversion) 

# Define a function to extract features from a single image window
def single_img_features(feature_params, img):    

    #1) Define an empty list to receive features
    img_features = []

    #2) Apply color conversion
    feature_image = BGR2_(img, feature_params.color_space)

    #3) Compute spatial features if flag is set
    if feature_params.spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=feature_params.spatial_size)

        #4) Append features to list
        img_features.append(spatial_features)

    #5) Compute histogram features if flag is set
    if feature_params.hist_feat == True:
        hist_features = color_hist(feature_image, nbins=feature_params.hist_bins)

        #6) Append features to list
        img_features.append(hist_features)

    #7) Compute HOG features if flag is set
    if feature_params.hog_feat == True:
        if feature_params.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    feature_params.orient, feature_params.pix_per_cell, feature_params.cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], feature_params.orient, 
                        feature_params.pix_per_cell, feature_params.cell_per_block, vis=False, feature_vec=True)

        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function to extract features from a list of images
def extract_features(feature_params, imgs):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        # Read in image
        image = cv2.imread(file)

        # Get features
        features.append(single_img_features(feature_params, image))

    # Return list of feature vectors
    return features
