import numpy as np
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from itertools import combinations
from scipy.spatial import ConvexHull
import numpy as np
import itertools
import scipy
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
# this folder have all the augmentation functions need for the dataaug

import numpy as np
import random

import matplotlib.pyplot as plt
import cv2
import math
import cv2
import math
import matplotlib as mpl
# import numpy as np
from bridson import poisson_disc_samples

from bridson import poisson_disc_samples

def create_mask_from_bbox(input_box, size=128):
    # Initialize an empty mask of size 128x128
    mask = np.zeros((size, size), dtype=np.uint8)

    # Extract the bounding box coordinates
    x_min, y_min, x_max, y_max = input_box

    # Ensure the bounding box coordinates are within the range
    x_min = max(0, min(size-1, x_min))
    y_min = max(0, min(size-1, y_min))
    x_max = max(0, min(size, x_max)) # x_max is exclusive
    y_max = max(0, min(size, y_max)) # y_max is exclusive

    # Set the region inside the bounding box to 1
    mask[y_min:y_max+1, x_min:x_max+1] = 255

    return mask

def visualize_and_save2(image, new_bbox_list, Center_Sets, radius, save_path):
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    radius = math.ceil(radius+1)

    # Mark original bounding box and its vertices in yellow and draw circles
    x_min, y_min, x_max, y_max = map(int, new_bbox_list[0])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)  # Yellow bounding box, the thickness == 1

    # Mark first point in Center_Sets with a yellow circle
    # x, y = map(int, Center_Sets[0]) #since center_sets doesn't contains the original points so...
    x, y = map(int, np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]))
    cv2.circle(image, (x, y), radius, (0, 255, 255), 1)  # Yellow circle, the -1 means the circle will be fullfilled

    # Mark rest of the points from Center_Sets in red
    for pt in Center_Sets[1:]:
        pt = tuple(map(int, pt))
        # cv2.circle(image, pt, radius, (0, 0, 255), -1)  # Red circle
        cv2.drawMarker(image, pt, (0, 0, 255), markerSize=1, thickness=1)  # red dot

    cv2.drawMarker(image, tuple(map(int, np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]))), (0, 255, 255), markerSize=2,
                   thickness=1)  # yellow dot for the center

    # Save image
    cv2.imwrite(save_path, image)

def visualize_and_save(image, new_bbox_list, VT1_set, VT2_set, VT3_set, VT4_set, radius, save_path1, save_path2):
    # Convert image to RGB if it's grayscale
    image2 = image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    radius = math.ceil(radius+1)
    ## The circle drawing in OpenCV might be slightly different from the method used to generate the points.
    # It's possible that the points are indeed within the circle according to the sampling method,
    # but the drawn circle appears slightly smaller due to how OpenCV interprets the radius and center point.

    # Mark original bounding box and its vertices in yellow and draw circles
    x_min, y_min, x_max, y_max = map(int, new_bbox_list[0])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)  # Yellow bounding box, the thickness == 1
    cv2.circle(image, (x_min, y_min), radius, (0, 255, 255), -1)  # Top-left, the -1 means the circle will be fullfilled
    cv2.circle(image, (x_max, y_min), radius, (0, 255, 255), -1)  # Top-right
    cv2.circle(image, (x_max, y_max), radius, (0, 255, 255), -1)  # Bottom-right
    cv2.circle(image, (x_min, y_max), radius, (0, 255, 255), -1)  # Bottom-left

    # Mark all sample points from VT sets in blue
    for VT_set in [VT1_set, VT2_set, VT3_set, VT4_set]:
        for pt in VT_set:
            pt = tuple(map(int, pt))
            # cv2.circle(image, pt, 1, (255, 0, 0), -1)  # Blue dot
            # cv2.drawMarker(image, pt, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)# Blue dot
            cv2.drawMarker(image, pt, (255, 0, 0), markerSize=1, thickness=1)  # Blue dot

    # Mark vertices of new_bbox_list[1:] in red
    for bbox in new_bbox_list[1:]:
        x_min, y_min, x_max, y_max = map(int, bbox)
        # cv2.circle(image, (x_min, y_min), 1, (0, 0, 255), -1)  # Top-left
        # cv2.circle(image, (x_max, y_min), 1, (0, 0, 255), -1)  # Top-right
        # cv2.circle(image, (x_max, y_max), 1, (0, 0, 255), -1)  # Bottom-right
        # cv2.circle(image, (x_min, y_max), 1, (0, 0, 255), -1)  # Bottom-left
        cv2.drawMarker(image, (x_min, y_min), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1)# Top-left
        cv2.drawMarker(image, (x_max, y_min), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=1,
                       thickness=1)  # Top-right
        cv2.drawMarker(image, (x_max, y_max), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=1,
                       thickness=1)  # Bottom-right
        cv2.drawMarker(image, (x_min, y_max), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=1,
                       thickness=1)  # Bottom-left
    # Save image
    cv2.imwrite(save_path1, image)

def sample_points3(input_box, M, N):
    # follows sample_points2, but using blue noise
    # If you still feel that the distribution of points is uneven, it may be because this sampling method is highly random, which may result in more points in some areas and fewer points in other areas. If you need even distribution,
    # You may want to use other sampling strategies, such as selecting points evenly on a grid within a circle, or using a technique called "blue noise". Blue noise is a special kind of noise that is uniform globally but maintains a certain degree of randomness locally.
    # Blue noise sampling is a sampling method that produces uniformly distributed points throughout the entire area while trying to maintain the minimum distance between adjacent points.
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = input_box

    # Calculate the center and side lengths
    C = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
    ES = min(x_max - x_min, y_max - y_min)

    # Calculate the radius
    radius = ES / M

    # Function to sample points from a circle
    def sample_points_from_circle(center, radius, num_points):
        points = poisson_disc_samples(width=2*radius, height=2*radius, r=radius/num_points)
        # Shift points to the right location
        points = [[point[0] + center[0] - radius, point[1] + center[1] - radius] for point in points]
        return np.array(points)

    # Sample N points from the circle
    Center_Sets = sample_points_from_circle(C, radius, N)

    # Generate new bounding boxes
    new_bbox_list = [input_box]
    for i in range(N):
        # Get new center point
        new_C = Center_Sets[i]

        # Calculate the new bounding box
        x_min_new = new_C[0] - (x_max - x_min) / 2
        y_min_new = new_C[1] - (y_max - y_min) / 2
        x_max_new = new_C[0] + (x_max - x_min) / 2
        y_max_new = new_C[1] + (y_max - y_min) / 2

        new_bbox = np.array([x_min_new, y_min_new, x_max_new, y_max_new])
        new_bbox_list.append(new_bbox)

    return new_bbox_list, radius, Center_Sets


def sample_points2(input_box, M, N):
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = input_box

    # Calculate the center and side lengths
    C = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
    ES = min(x_max - x_min, y_max - y_min)

    # Calculate the radius
    radius = ES / M

    # Function to sample points from a circle
    def sample_points_from_circle(center, radius, num_points):
        points = []
        for _ in range(num_points):
            # Sample a point in polar coordinates and convert it to cartesian
            theta = 2 * np.pi * random.random()
            r = radius * np.sqrt(random.random())
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    # Sample N points from the circle
    Center_Sets = sample_points_from_circle(C, radius, N)

    # Generate new bounding boxes
    new_bbox_list = [input_box]
    for i in range(N):
        # Get new center point
        new_C = Center_Sets[i]

        # Calculate the new bounding box
        x_min_new = new_C[0] - (x_max - x_min) / 2
        y_min_new = new_C[1] - (y_max - y_min) / 2
        x_max_new = new_C[0] + (x_max - x_min) / 2
        y_max_new = new_C[1] + (y_max - y_min) / 2

        new_bbox = np.array([x_min_new, y_min_new, x_max_new, y_max_new])
        new_bbox_list.append(new_bbox)

    return new_bbox_list, radius, Center_Sets


def sample_points(input_box, M, N, S, sample_type = "MC"):
#  2) scale ratio M, 3) number of sampled bounding box: N, 4) number of sampled_points S
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = input_box

    # Calculate the vertices and side lengths
    VT1 = np.array([x_min, y_min])
    VT2 = np.array([x_max, y_min])
    VT3 = np.array([x_max, y_max])
    VT4 = np.array([x_min, y_max])
    ES = min(x_max - x_min, y_max - y_min)

    # Sample S points for each vertex
    def sample_points_from_circle(center, radius, num_points):
        points = []
        for _ in range(num_points):
            # Sample a point in polar coordinates and convert it to cartesian
            theta = 2 * np.pi * random.random()
            r = radius * np.sqrt(random.random())
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    def sample_points_from_gaussian(center, radius, num_points):
        # The mean would be the center of the circle, and the covariance matrix would be a 2D identity matrix scaled
        # by the desired variance (which could be set to (radius**2)/2 to ensure that approximately 95% of points fall
        # within the circle).

        # a Gaussian distribution is not bounded, so it's still possible to get points outside the circle.


        # Set the mean and covariance
        mean = center
        covariance = np.eye(2) * (radius ** 2) / 2

        # Sample points from the Gaussian distribution
        points = np.random.multivariate_normal(mean, covariance, num_points)

        return points

    radius = ES / M

    if sample_type == 'MC':
        VT1_set = sample_points_from_circle(VT1, radius, S)
        VT2_set = sample_points_from_circle(VT2, radius, S)
        VT3_set = sample_points_from_circle(VT3, radius, S)
        VT4_set = sample_points_from_circle(VT4, radius, S)
    else:
        VT1_set = sample_points_from_gaussian(VT1, radius, S)
        VT2_set = sample_points_from_gaussian(VT2, radius, S)
        VT3_set = sample_points_from_gaussian(VT3, radius, S)
        VT4_set = sample_points_from_gaussian(VT4, radius, S)

    # Generate new bounding boxes
    new_bbox_list = [input_box]
    for _ in range(N):
        # Sample a point from each set
        new_VT1 = VT1_set[random.randint(0, S-1)]
        new_VT2 = VT2_set[random.randint(0, S-1)]
        new_VT3 = VT3_set[random.randint(0, S-1)]
        new_VT4 = VT4_set[random.randint(0, S-1)]

        # Calculate the new bounding box
        x_min_new = min(new_VT1[0], new_VT4[0])
        y_min_new = min(new_VT1[1], new_VT2[1])
        x_max_new = max(new_VT2[0], new_VT3[0])
        y_max_new = max(new_VT3[1], new_VT4[1])

        new_bbox = np.array([x_min_new, y_min_new, x_max_new, y_max_new])
        new_bbox_list.append(new_bbox)

    return new_bbox_list, radius, VT1_set, VT2_set, VT3_set, VT4_set


def calculate_aleatoric_uncertainty(mask_list):
    # stack the mask samples in the mask_list to calculate the frequency of each pixel
    all_masks = np.vstack(mask_list)

    # calculate the frequency of 1 for each pixel location
    frequency = np.mean(all_masks, axis=0)

    # calculate the aleatoric uncertainty for each frequency location
    aleatoric_uncertainty = frequency * (1 - frequency)

    # change to an entrophy type:
    # aleatoric_uncertainty = -0.5*(frequency * np.log(frequency + 10e-7)+(1 - frequency) * np.log((1 - frequency) + 10e-7))

    return aleatoric_uncertainty


def visualize_uncertainty2(uncertainty_map):
    plt.imshow(uncertainty_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Aleatoric Uncertainty Map')
    plt.show()

def show_uncertainty(uncertainty_map, ax):
    ax.imshow(uncertainty_map,cmap='hot', interpolation='nearest')

def overlay_uncertainty_on_image(image, uncertainty_map, ax):
    """
    Superimpose the uncertainty map on the image and display it.

     parameter:
     image (numpy.ndarray): Input image, a single-channel image of shape (1, height, width).
     uncertainty_map (numpy.ndarray): uncertainty map, shape (height, width).

     return:
     There is no return value, and the superimposed image is displayed directly.

    """
    image_rgb = image

    # Create a colormap from white to red
    cmap = plt.get_cmap('hot')
    # cmap.set_under('red')  # sets color for smallest values
    # cmap.set_over('white')  # sets color for largest values

    # Map the colors in the colormap to the range 0-0.25
    norm = mpl.colors.Normalize(vmin=0, vmax=0.25)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    uncertainty_map_rgb = cmap(norm(uncertainty_map))

    uncertainty_map_rgb = (uncertainty_map_rgb[..., :3] * 255).astype(np.uint8)  # ignore alpha channel

    # overlay the uncertainty map on the img
    overlay_image = np.copy(image_rgb)
    overlay_image[..., 0] = (overlay_image[..., 0] * 0.7 + uncertainty_map_rgb[..., 0] * 0.3)
    overlay_image[..., 1] = (overlay_image[..., 1] * 0.7 + uncertainty_map_rgb[..., 1] * 0.3)
    overlay_image[..., 2] = (overlay_image[..., 2] * 0.7 + uncertainty_map_rgb[..., 2] * 0.3)

    # Show superimposed images
    im = ax.imshow(overlay_image)

    #  colorbar
    plt.colorbar(mapper, ax=ax, extend='neither') # set extend from 'both' to 'neither' to remove the arrow in the colorbar



def simple_threshold(aleatoric_uncertainty_map , uncertainty_threshold = 0.5):
    # thresholding to get final mask
    final_mask = np.where(aleatoric_uncertainty_map <= uncertainty_threshold, 1, 0)

    return final_mask


def mask_adjustment(binary_mask, uncertainty_map, threshold_ratio = 0.2):
    # Get the threshold
    threshold = threshold_ratio * np.max(uncertainty_map)

    # Identify the pixels in the uncertainty map that are above the threshold
    above_threshold = uncertainty_map > threshold

    # Apply the condition to the binary mask
    binary_mask[0, above_threshold] = 0

    return binary_mask

def mask_adjustment2(binary_mask, uncertainty_map, img, threshold_ratio = 0.2, thre_fp = 0.5, thre_fn=0.5 ):
    # Get the threshold
    threshold = threshold_ratio * np.max(uncertainty_map)

    # Identify the pixels in the uncertainty map that are above the threshold
    above_threshold = uncertainty_map > threshold

    # Average over the channels of img
    img_avg = np.mean(img, axis=2)

    # Calculate the average value of img under the binary_mask
    mean_value = np.mean(img_avg[binary_mask[0] > 0])

    # For the binary mask,
    # if the pixels are in the above_threshold and the corresponding positional pixel in img larger than mean_value, set it to be 0,
    # binary_mask[0, (above_threshold) & (img_avg > mean_value*thre_fp)] = 0
    binary_mask[0, above_threshold] = 0

    # if the pixels are in the above_threshold and the corresponding positional pixel in img smaller or equal to mean_value, set to be 1
    binary_mask[0, (above_threshold) & (img_avg <= mean_value* thre_fn)] = 1

    return binary_mask


def mask_adjustment3(binary_mask, uncertainty_map, img, threshold_ratio = 0.2, thre_fp = 0.5, thre_fn=0.5 ):
    # Get the threshold
    threshold = threshold_ratio * np.max(uncertainty_map)

    # Identify the pixels in the uncertainty map that are above the threshold
    above_threshold = uncertainty_map > threshold

    # Average over the channels of img
    img_avg = np.mean(img, axis=2)


    # For the binary mask,
    # if the pixels are in the above_threshold and the corresponding positional pixel in img larger than mean_value, set it to be 0,
    # binary_mask[0, (above_threshold) & (img_avg > mean_value*thre_fp)] = 0
    binary_mask[0, above_threshold] = 0

    # Calculate the average value of img under the binary_mask, after threshold by uncertainty map
    mean_value = np.mean(img_avg[binary_mask[0] > 0])

    # if the pixels are in the above_threshold and the corresponding positional pixel in img smaller or equal to mean_value, set to be 1
    binary_mask[0, (above_threshold) & (img_avg <= mean_value* thre_fn)] = 1

    return binary_mask

def mask_adjustment4(binary_mask, uncertainty_map, img, threshold_ratio = 0.2, thre_fp = 0.5, thre_fn=0.5 ):
    # Get the threshold
    threshold = np.min(uncertainty_map)+threshold_ratio * (np.max(uncertainty_map) - np.min(uncertainty_map))

    # Identify the pixels in the uncertainty map that are above the threshold
    above_threshold = uncertainty_map > threshold

    original_binary_range = binary_mask>0

    # Average over the channels of img
    img_avg = np.mean(img, axis=2)


    # For the binary mask,
    # if the pixels are in the above_threshold and the corresponding positional pixel in img larger than mean_value, set it to be 0,
    # binary_mask[0, (above_threshold) & (img_avg > mean_value*thre_fp)] = 0
    binary_mask[0, above_threshold] = 0

    # Calculate the average value of img under the binary_mask, after threshold by uncertainty map
    mean_value = np.mean(img_avg[binary_mask[0] > 0])

    # print("original_binary_rang shape:", original_binary_range.shape) # (1, 256, 256)
    # print("above_threshold shpae:", above_threshold.shape) # (256, 256)
    # print("img_avg <= mean_value* thre_fn:", (img_avg <= mean_value* thre_fn).shape) # (256, 256)

    # if the pixels are in the above_threshold and the corresponding positional pixel in img smaller or equal to mean_value, set to be 1
    binary_mask[0,(original_binary_range[0])&(above_threshold) & (img_avg <= mean_value* thre_fn)] = 1

    return binary_mask