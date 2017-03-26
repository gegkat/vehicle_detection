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

class WindowParams():
    def __init__(self, ylims=(None, None), scale=1.0, xy_skips=(1,1)):
        self.ylims = ylims
        self.scale = scale
        self.xy_skips = xy_skips

class VehicleDetector():
    def __init__(self, feature_params, map_lims=(-12,8)):
        self.svc, self.X_scaler = load_svc(fname="svc.p")
        self.feature_params = feature_params
        self.MAP_MIN = map_lims[0]
        self.MAP_MAX = map_lims[1]
        self.HEATMAP_THRESHOLD = 0
        self.heatmap = None

        self.window_params_list = []
        self.window_params_list.append(WindowParams(ylims=(380,620), scale=2.0, xy_skips=(1,1)))
        self.window_params_list.append(WindowParams(ylims=(390,550), scale=1.5, xy_skips=(2,2)))
        self.window_params_list.append(WindowParams(ylims=(410,540), scale=1.2, xy_skips=(1,1)))


    def process_img(self, img):
        if self.heatmap is None:
            self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

        curr_heat = np.zeros_like(img[:,:,0]).astype(np.float)

        for window_params in self.window_params_list:
            img_out1, on_windows = find_cars(img, self.feature_params, window_params, self.svc, self.X_scaler)
            curr_heat = add_heat(curr_heat, on_windows)

        self.heatmap += curr_heat-1

        self.heatmap[self.heatmap < self.MAP_MIN] = self.MAP_MIN
        self.heatmap[self.heatmap > self.MAP_MAX] = self.MAP_MAX


        heat_thresh = apply_threshold(self.heatmap, self.HEATMAP_THRESHOLD)
        labels = label(heat_thresh)
        img_out = draw_labeled_bboxes(img, labels)

        tmp = np.copy(self.heatmap)
        tmp = (tmp - self.MAP_MIN)*255/(self.MAP_MAX - self.MAP_MIN)
        tmp *= 0.8

        img_out2 = np.zeros_like(img_out)
        img_out2[:,:,0] = tmp
        img_out2[:,:,1] = heat_thresh*255/np.max(heat_thresh)

        return img_out, img_out2



