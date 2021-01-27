import numpy as np
from numpy.linalg import norm

def calculate_contrast(img):
    # Contrast = sqrt(sum(I - I_bar)^2 / (M*N) )
    # See https://en.wikipedia.org/wiki/Contrast_(vision)#RMS_contrast
    if np.max(img) > 1:
        img = np.divide(img, [255, 255, 255])
    c = norm(img - np.mean(img, axis=(0, 1))) / np.sqrt(img.shape[0] * img.shape[1])
    return c

def scale_img_contrast(img):
    """
    Increase the contrast of an image by scaling its min and max.
    Returns a value between 0 and 1.
    """
    i_min = np.min(img, axis=(0, 1))
    i_max = np.max(img, axis=(0, 1))
    scaled = (img - i_min) / (i_max - i_min)
    return scaled

# Not used
def brighten(img, yen_thresholding=True, unit_range=True):
    """
    After contrast has been increased, it can be boosted further by thresholding.
    If `yen_thresholding` is False, 2nd and 98th percentiles are used instead.
    """
    if yen_thresholding:
        yen_threshold = threshold_yen(img)
        in_range = (0, yen_threshold)
    else:
        in_range = np.percentile(img, (2, 98))
    if unit_range:
        min_max = (0, 1)
    else:
        min_max = (0, 255)
    brightened = rescale_intensity(img, in_range, min_max)
    return scale_img(brightened)
