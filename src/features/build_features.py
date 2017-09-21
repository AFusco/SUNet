import numpy as np


def int32_to_uint8(matrix):
    """Convert a uint16 numpy array to a uint8 numpy array"""
    if matrix.dtype != np.int32:
        raise ValueError("Given matrix was expected to be of type np.uint16, but is " + matrix.dtype)

    return (matrix/256).astype(np.uint8)

def threshold_confidence(disparity, disparity_gt, threshold):
    """Create a confidence map from a disparity map and a groundtruth disparity map.

    The confidence map is created by comparing the difference between the input maps with the threshold.
    Each cell in the output map may assume one of the following values:
        0 if disparity[x,y] - disparity_gt[x,y] > threshold
        1 if disparity[x,y] - disparity_gt[x,y] <= threshold
        -1 if disparity_gt[x,y] == 0, indicating that a groundtruth value is not available

    Args:
        disparity: The disparity map
        disparity_gt: The groundtruth disparity map. It may contain zero values indicating
            that a groundtruth is not available for a given pixel. In that case, the pixel
            will be set to -1 in the resulting map.

            The two images must match both in shape and datatype.

        threshold: The confidence threshold for the disparity map.

    Returns:
        A numpy array of floats, of same shape as the input.
    """

    if disparity.shape != disparity_gt.shape:
        raise ValueError("The maps shapes must match!")

    if disparity.dtype != disparity_gt.dtype:
        raise ValueError("The maps types must match!")

    
    # Wrong pixels (it may contain unknown pixels)
    wrong_mask = np.absolute( disparity - disparity_gt ) > threshold

    # Unkown pixels are the pixels in the dmap_gt that are set to 0
    unknown_mask = disparity_gt == 0

    conf_gt = np.ones_like(disparity, dtype=np.float32) # Initialize everything to 1
    conf_gt[wrong_mask] = 0
    conf_gt[unknown_mask] = -1

    return conf_gt



def composite_confidence(disparity, disparity_gt, max_threshold):
    """Create a confidence map from a disparity map and a groundtruth disparity map.

    The confidence map is created by combining all the threshold confidence maps with all the confidences between 0 and max_threshold.

    Each cell in the output map may assume one of the following values:
        [0,1] indicating the probability of a pixel being wrong
        -1 if disparity_gt[x,y] == 0, indicating that a groundtruth value is not available

    Args:
        disparity: The disparity map
        disparity_gt: The groundtruth disparity map. It may contain zero values indicating
            that a groundtruth is not available for a given pixel. In that case, the pixel
            will be set to -1 in the resulting map.

            The two images must match both in shape and datatype.

        threshold: The confidence threshold for the disparity map.

    Returns:
        A numpy array of floats, of same shape as the input.
    """
    # Unkown pixels are the pixels in the dmap_gt that are set to 0
    wrong_mask = np.absolute( disparity - disparity_gt ) > threshold
    unknown_mask = disparity_gt == 0

    conf_gt = np.ones_like(disparity, dtype=np.float32)
    conf_gt[wrong_mask] = 0
    conf_gt[unknown_mask] = -1

    return conf_gt

def difference(disparity, disparity_gt):
    raise NotImplementedError
    unknown_mask = disparity_gt == 0
    diff = np.absolute(disparity - disparity_gt/256.0) / 256.0
    diff[unknown_mask] = 0.5
    return diff
