import numpy as np
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.spatial import distance

# def assd(mask1, mask2):
#     # Get the set of non-zero points, representing the surface
#     surface_mask1 = np.argwhere(mask1)
#     surface_mask2 = np.argwhere(mask2)
#
#     # Compute distances from surface_mask1 to surface_mask2 and vice versa
#     forward_distance = np.mean([np.min(distance.cdist(surface_mask1[i:i+1], surface_mask2, 'euclidean')) for i in range(surface_mask1.shape[0])])
#     backward_distance = np.mean([np.min(distance.cdist(surface_mask2[i:i+1], surface_mask1, 'euclidean')) for i in range(surface_mask2.shape[0])])
#
#     # Compute the symmetric distance
#     symmetric_distance = np.mean([forward_distance, backward_distance])
#
#     return symmetric_distance

from scipy.spatial import distance
import numpy as np

def assd(mask1, mask2):
    # Check if mask1 or mask2 is all zero
    if np.all(mask1 == 0) or np.all(mask2 == 0):
        return 1

    # Get the set of non-zero points, representing the surface
    surface_mask1 = np.argwhere(mask1)
    surface_mask2 = np.argwhere(mask2)

    # Compute distances from surface_mask1 to surface_mask2 and vice versa
    forward_distance = np.mean([np.min(distance.cdist(surface_mask1[i:i+1], surface_mask2, 'euclidean')) for i in range(surface_mask1.shape[0])])
    backward_distance = np.mean([np.min(distance.cdist(surface_mask2[i:i+1], surface_mask1, 'euclidean')) for i in range(surface_mask2.shape[0])])

    # Compute the symmetric distance
    symmetric_distance = np.mean([forward_distance, backward_distance])

    return symmetric_distance




def dice_score(mask1, mask2):
    # Flatten the masks
    mask1 = mask1.flatten()
    mask2 = mask2.flatten()

    # Compute Dice coefficient
    intersect = np.sum(mask1 * mask2)
    return (2. * intersect) / (np.sum(mask1) + np.sum(mask2))

def hausdorff_distance(mask1, mask2):
    # Compute Hausdorff distance
    return max(directed_hausdorff(mask1, mask2)[0], directed_hausdorff(mask2, mask1)[0])


def calculate_fp_fn(mask_gt, mask_pred):
    """
    Calculate FP and FN rates for binary masks.

    Parameters:
    mask_gt (numpy.ndarray): Ground truth binary mask, shape (D, D).
    mask_pred (numpy.ndarray): Predicted binary mask, shape (D, D).

    Returns:
    float, float: FP rate, FN rate
    """
    assert mask_gt.shape == mask_pred.shape, "Shape mismatch between ground truth and predicted masks."

    # Flatten the masks and combine
    labels = 2 * mask_gt.reshape(-1) + mask_pred.reshape(-1)

    # Count occurrences of 00, 01, 10, 11
    counts = np.bincount(labels, minlength=4)

    TN, FP, FN, TP = counts

    # Calculate FP and FN rates
    fp_rate = FP / (FP + TN) if FP + TN != 0 else 0
    fn_rate = FN / (FN + TP) if FN + TP != 0 else 0

    return fp_rate, fn_rate
