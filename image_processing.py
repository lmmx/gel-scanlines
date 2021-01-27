from imageio import imread, imwrite
from scipy.fftpack import fft2
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

def grade(img, make_grey=True, make_uint8=True, sigma=None):
    """
    Set up an image for gradient calculation.
    Assumes image is either scaled at 0-1 or 0-255, and will
    convert to uint8 by scaling up to 255 if necessary.
    """
    if sigma is None:
        sigma = 0.4
    if make_grey:
        img = rgb2grey(img)
    if make_uint8 and img.dtype != "uint8":
        if np.max(img) == 1.0:
            img *= 255
        img = np.uint8(img)
    graded = auto_canny(img, sigma=sigma)
    return graded

def plot_fft_spectrum(img, prune_percentile=95):
    """
    Plot the FFT spectrum of the image, along with a high contrast version,
    and then along with a pruned version of this high contrast spectrum, in which
    only the values above the bottom {prune_percentile} are kept (e.g. at 95%,
    only the top 5%ile of values is displayed).
    Note that the image passed in should be the brightened/boosted/scaled Canny
    gradient image output, not an unmodified photo.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    imf = fft2(img)
    ax1.imshow(np.abs(imf), norm=LogNorm(vmin=5))
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(img)
    ax3 = fig.add_subplot(3, 1, 3)
    mod_log_maxima = prune_fft(img, prune_percentile=prune_percentile)
    ax3.imshow(mod_log_maxima)
    plt.show()
    return
