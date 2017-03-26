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

# Class for holding parameters that define window search
class WindowParams():
    def __init__(self, ylims=(None, None), scale=1.0, xy_skips=(1,1)):
        self.ylims = ylims # crops image between ylims[0] and ylims[1]
        self.scale = scale # scales image
        self.xy_skips = xy_skips # skips HOG steps

# Class for applying a trained svc to a video stream or image
class VehicleDetector():
    def __init__(self, feature_params, map_lims=(-12,8), heatmap_threshold=0):
        # Load trained svc
        self.svc, self.X_scaler = load_svc(fname="svc.p")

        # Log feature parameters
        self.feature_params = feature_params

        # Constants
        self.MAP_MIN = map_lims[0]
        self.MAP_MAX = map_lims[1]
        self.HEATMAP_THRESHOLD = heatmap_threshold

        # Initialize heatmap to None
        self.heatmap = None

        # List of WindoParams objects for defining search windows at different scales
        self.window_params_list = []
        self.window_params_list.append(WindowParams(ylims=(380,610), scale=1.8, xy_skips=(1,1)))
        self.window_params_list.append(WindowParams(ylims=(400,520), scale=1.2, xy_skips=(2,1)))

        # For plotting only
        self.on_windows_list = []
        self.all_windows_list = []
        self.orig_img = None


    # Function to ingest a new image, update heatmap, and determine vehicle positions
    def process_img(self, img):
        
        # Log the original image for plotting purposes
        self.orig_img = np.copy(img)

        # If the heatmap has not been initialized, set it to all 0
        # This will only happen on the first image in the stream
        if self.heatmap is None:
            self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

        # Initialize current heat map and window lists
        self.curr_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        self.on_windows_list = []
        self.all_windows_list = []

        # Loop through each definied window scaling and find_cars 
        for window_params in self.window_params_list:
            on_windows, all_windows = find_cars(img, self.feature_params, window_params, self.svc, self.X_scaler)

            # Log for optional plotting 
            self.on_windows_list.append(on_windows)
            self.all_windows_list.append(all_windows)

            # Increment the heat map in all windows with a car found
            self.curr_heatmap = add_heat(self.curr_heatmap, on_windows)


        # The timeseries heatmap keeps track of the confidence that there is a car
        # at each index. At each timestep we decrement the heatmap by 1 and then 
        # increment it by the curr_heatmap. That way we are combining previous 
        # knowledge of where cars were detected with new knowledge
        self.heatmap += -1
        self.heatmap += self.curr_heatmap

        # Saturate the heatmap to limit the amount of time to respond
        # to new information about car locations
        self.heatmap[self.heatmap < self.MAP_MIN] = self.MAP_MIN
        self.heatmap[self.heatmap > self.MAP_MAX] = self.MAP_MAX

        # Threshold the timeseries heatmap
        self.heatmap_thresh = apply_threshold(self.heatmap, self.HEATMAP_THRESHOLD)

        # Label the thresholded heatmap
        self.labels = label(self.heatmap_thresh)

        # Draw boxes around labels
        self.img_out = draw_labeled_bboxes(img, self.labels)

        # BGR2RGB
        self.img_out = BGR2_(self.img_out, 'RGB')

    # Function to generate images for plotting
    def plot_images(self):
        draw_img_all = np.copy(self.orig_img)
        draw_img_all = BGR2_(draw_img_all, 'RGB')
        colors = [(255,0,0), (0, 255, 0), (255, 255, 0)]
        count = 0
        self.draw_img_all_list = []
        for i in range(len(self.all_windows_list)):
            draw_img_all = np.copy(self.orig_img)
            for rect in self.all_windows_list[i]:
                cv2.rectangle(draw_img_all, rect[0], rect[1], colors[i], 6)

            for rect in self.on_windows_list[i]:
                cv2.rectangle(draw_img_all, rect[0], rect[1], (0,0,255), 6)

            self.draw_img_all_list.append(BGR2_(draw_img_all, 'RGB'))
  

        draw_img_found = np.copy(self.orig_img)
        for windows in self.on_windows_list:
            for rect in windows:
                cv2.rectangle(draw_img_found, rect[0], rect[1],(0,0,255),6) 
        self.draw_img_found = BGR2_(draw_img_found, 'RGB')

  
        heat_img = np.zeros_like(self.orig_img)
        heat_img[:,:,0] = self.curr_heatmap*255/10
        heat_img[:,:,1] = self.curr_heatmap*255/10
        heat_img[:,:,2] = self.curr_heatmap*255/10

        heat_img = draw_labeled_bboxes(heat_img, self.labels)

        self.heat_img = BGR2_(heat_img, 'RGB')


        heat_thresh_img = np.zeros_like(self.orig_img)
        heat_thresh = np.copy(self.heatmap_thresh)
        heat_thresh[heat_thresh > 1] = 1
        heat_thresh_img[:,:,0] = heat_thresh*255
        heat_thresh_img[:,:,1] = heat_thresh*255
        heat_thresh_img[:,:,2] = heat_thresh*255
        heat_thresh_img = draw_labeled_bboxes(heat_thresh_img, self.labels)

        self.heat_thresh_img = BGR2_(heat_thresh_img, 'RGB')
        

        heatmap_scaled = np.copy(self.heatmap)
        heatmap_scaled = (heatmap_scaled - self.MAP_MIN)*255/(self.MAP_MAX - self.MAP_MIN)
        heatmap_scaled *= 0.8

        heatmap_img_out = np.zeros_like(self.orig_img)
        heatmap_img_out[:,:,2] = heatmap_scaled
        # heatmap_img_out[:,:,1] = self.heatmap*255/np.max(self.heatmap)

        heatmap_img_out = BGR2_(heatmap_img_out, 'RGB')

        heatmap_img_out = cv2.addWeighted(heatmap_img_out, 1, self.img_out, 0.5, 0)
        self.heatmap_img_out = heatmap_img_out


    # Function to plot all plot images
    def plot(self):
        f,axs = plt.subplots(2,3, figsize=(20,12))
        for i in range(len(self.all_windows_list)):
            axs[0, i].imshow(self.draw_img_all_list[i])
        axs[0, 2].imshow(self.draw_img_found)
        axs[1, 0].imshow(self.heat_img)
        axs[1, 1].imshow(self.heat_thresh_img)
        axs[1, 2].imshow(self.img_out)
        plt.show()


