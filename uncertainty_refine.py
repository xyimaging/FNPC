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
# import numpy as np
from bridson import poisson_disc_samples

from bridson import poisson_disc_samples


# the uncertainty-based FNPC block in hardcode version
def uc_refine_hardcode(binary_mask, uncertainty_map, img, threshold_uc = 0.2, fn_alpha = 0.5, fn_beta = 0.7,fp_alpha = 0.5, fp_beta = 0.7 ):
    # based on uc_refine but don't use mean value, use the range directly
    # binary_mask: 1xDxD (0, 1), the pseudo label
    # uncertainty_map: DxD (0 - 0.25)
    # img: DxDx3 (0, 255), the original image

    # Average over the channels of img
    img_avg = np.mean(img, axis=2)

    # calculate the mean xt(yt)
    mean_value = np.mean(img_avg * binary_mask[0])
    print("the mean value is:", mean_value)

    # Get the uncertainty threshold value
    uc_threshold = np.min(uncertainty_map)+threshold_uc * (np.max(uncertainty_map) - np.min(uncertainty_map))

    # Identify the U_thre (turn uc to a binary mask)
    U_thre = uncertainty_map > uc_threshold

################## do the FN correction ###################
    # get the 1 - yt
    # We know that the region outside the bbox should be true negative, so here should be written
    # as bbox - binary_mask. But don't worry, the current unperfect version doesn't affect the final results, since we
    # refine the final mask by BBox in the FNPC_Interface.py
    inverse_binary_mask = 1 - binary_mask # get the region where is not covered by pseudo mask

    # get the UH = (1-yt)*U_thre
    FN_UH = U_thre * inverse_binary_mask[0] # DxD

    # get the xUH
    FN_xUH = img_avg * FN_UH # DxD

    # pseudo label FN refine
    # Create a boolean mask where True indicates the condition is met
    FN_condition_mask = (fn_alpha < FN_xUH) & (FN_xUH < fn_beta)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FN_new_mask = np.logical_or(binary_mask, FN_condition_mask) # the return is DxD since 1xDxD will be consider as DxD in np logical
    FN_output_mask = FN_new_mask# 1xDxD

    ################## do the FP correction ###################
    FP_UH = U_thre * binary_mask[0]  # DxD

    # get the xUH
    FP_xUH = img_avg * FP_UH  # DxD
    # print(FP_xUH)

    # pseudo label FP refine
    # Create a boolean mask where True indicates the condition is met
    FP_condition_mask = ((fp_alpha > FP_xUH) | (FP_xUH >fp_beta))&(FP_UH>0)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FP_new_mask = np.logical_and(FN_output_mask, np.logical_not(
        FP_condition_mask))  # the return is DxD since 1xDxD will be consider as DxD in np logical

    # output_mask = np.expand_dims(new_mask, axis=0)
    FP_output_mask = FP_new_mask.astype(int)  # 1xDxD
    return FN_output_mask, FN_UH, FN_xUH, FN_condition_mask, FP_output_mask, FP_UH, FP_xUH, FP_condition_mask

# the uncertainty-based FNPC block in ratio version
def uc_refine_correct(binary_mask, uncertainty_map, img, threshold_uc = 0.2, fn_alpha = 0.5, fn_beta = 0.7,fp_alpha = 0.5, fp_beta = 0.7 ):
    # binary_mask: 1xDxD (0, 1), the pseudo label
    # uncertainty_map: DxD (0 - 0.25)
    # img: DxDx3 (0, 255), the original image

    # Average over the channels of img
    img_avg = np.mean(img, axis=2)

    # calculate the mean xt(yt)
    mean_value = np.mean(img_avg[(binary_mask[0] > 0)])  # for the placenta task

    print("the mean value is:", mean_value)

    # Get the uncertainty threshold value
    uc_threshold = np.min(uncertainty_map)+threshold_uc * (np.max(uncertainty_map) - np.min(uncertainty_map))

    # Identify the U_thre (turn uc to a binary mask)
    U_thre = uncertainty_map > uc_threshold

################## do the FN correction ###################
    # get the 1 - yt
    # We know that the region outside the bbox should be true negative, so here should be written
    # as bbox - binary_mask. But don't worry, the current unperfect version doesn't affect the final results, since we
    # refine the final mask by BBox in the FNPC_Interface.py
    inverse_binary_mask = 1 - binary_mask # get the region where is not covered by pseudo mask

    # get the UH = (1-yt)*U_thre
    FN_UH = U_thre * inverse_binary_mask[0] # DxD

    # get the xUH
    FN_xUH = img_avg * FN_UH # DxD

    # pseudo label FN refine
    # Create a boolean mask where True indicates the condition is met
    FN_condition_mask = (mean_value * fn_alpha < FN_xUH) & (FN_xUH < mean_value * fn_beta)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FN_new_mask = np.logical_or(binary_mask, FN_condition_mask) # the return is DxD since 1xDxD will be consider as DxD in np logical

    # output_mask = np.expand_dims(new_mask, axis=0)
    FN_output_mask = FN_new_mask# 1xDxD

    ################## do the FP correction ###################
    FP_UH = U_thre * binary_mask[0]  # DxD

    # get the xUH
    FP_xUH = img_avg * FP_UH  # DxD

    # pseudo label FP refine
    # Create a boolean mask where True indicates the condition is met
    FP_condition_mask = ((mean_value * fp_alpha > FP_xUH) | (FP_xUH > mean_value * fp_beta))&(FP_UH>0)
    # FP_condition_mask =  (FP_xUH > mean_value * fp_beta)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FP_new_mask = np.logical_and(FN_output_mask, np.logical_not(
        FP_condition_mask))  # the return is DxD since 1xDxD will be consider as DxD in np logical

    # output_mask = np.expand_dims(new_mask, axis=0)
    FP_output_mask = FP_new_mask.astype(int)  # 1xDxD
    return FN_output_mask, FN_UH, FN_xUH, FN_condition_mask, FP_output_mask, FP_UH, FP_xUH, FP_condition_mask