from imageio import imread, imwrite
from matplotlib import pyplot as plt
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from pathlib import Path

from image_processing import calculate_contrast, scale_img_contrast, brighten, grade

SAVING = True

gel_img_filename = Path("hannah-thomas-gel-photo.jpeg")
example_filenames = [f"{gel_img_filename.stem}_example_t{n}{gel_img_filename.suffix}" for n in range(1,5)]

gel_img = scale_img_contrast(rgb2gray(imread(gel_img_filename)))
eg_images = [scale_img_contrast(rgb2gray(imread(im))) for im in example_filenames] # 4 example progress images

first_row = np.argwhere(gel_img.max(axis=1) > 0).ravel()[0]
#last_row = np.argwhere(gel_img[first_row:,:].max(axis=1) == 0).ravel()[0] # argmin
last_row = np.argmin(gel_img[first_row:,:].max(axis=1))
cropped = gel_img[first_row:first_row+last_row, :]
cropped_eg_images = [im[first_row:first_row+last_row, :] for im in eg_images]
all_gel_imgs = (cropped, *cropped_eg_images)
crop_1, crop_2, crop_3, crop_4 = cropped_eg_images

# Obtain (row, col) coords for the gel from manual inspection in matplotlib
start_row, start_col = 400, 480
end_row, end_col = 466, 640

all_images = cropped, crop_1, crop_2, crop_3, crop_4

all_images = [
    im[start_row:end_row, start_col:end_col] for im in all_images
]

cropped, crop_1, crop_2, crop_3, crop_4 = all_images 

dc0 = np.zeros_like(cropped, dtype=float)
dc1 = np.abs(np.clip(crop_1 - cropped, a_min=None, a_max=0.))
dc2 = np.abs(np.clip(crop_2 - crop_1, a_min=None, a_max=0.))
dc3 = np.abs(np.clip(crop_3 - crop_2, a_min=None, a_max=0.))
dc4 = np.abs(np.clip(crop_4 - crop_3, a_min=None, a_max=0.))

all_diffs = [dc0, dc1, dc2, dc3, dc4]

def clip_percentiles(scanline_img, lower=85, upper=100):
    lower_clip, upper_clip = np.percentile(scanline_img[0,:], [lower, upper])
    return np.clip(scanline_img, a_min=lower_clip, a_max=upper_clip)

def make_scanlines(im, clipping=True):
    "Take vertical scanlines (column maxima)"
    scanned = im.copy()
    col_max = scanned.T.max(axis=1).reshape(-1,1)
    scanlines = np.repeat(col_max, [scanned.shape[0]], axis=1).T
    if clipping:
        scanlines = clip_percentiles(scanlines, lower=50)
    return scanlines

sc0, sc1, sc2, sc3, sc4 = [make_scanlines(im) for im in all_diffs]
all_scanlines = sc0, sc1, sc2, sc3, sc4
all_scanlines_unclipped = [sc.copy() for sc in all_scanlines]
su0, su1, su2, su3, su4 = all_scanlines_unclipped

am0, am1, am2, am3, am4 = [sc[0,:].argmax() for sc in all_scanlines]
all_sc_am = am0, am1, am2, am3, am4 

for sc, am in zip(all_scanlines[2:], all_sc_am[1:-1]):
    #max_of_frontier = sc[:, :am].max()
    sc[:, am:] = 0 #np.multiply(0.99, max_of_frontier)

fc0, fc1, fc2, fc3, fc4 = [
    im for im in all_scanlines
]

fig, (
    (ax0, ax1, ax2, ax3, ax4),
    (ax0d, ax1d, ax2d, ax3d, ax4d),
    (ax0s, ax1s, ax2s, ax3s, ax4s),
    (ax0b, ax1b, ax2b, ax3b, ax4b)
) = plt.subplots(4,5, sharex=True, sharey=True)
plt.subplots_adjust(hspace=-0.85)
ax0.imshow(cropped)
ax1.imshow(crop_1)
ax2.imshow(crop_2)
ax3.imshow(crop_3)
ax4.imshow(crop_4)
ax0.set_title("Time step 0", usetex=True)
ax1.set_title("Time step 1", usetex=True)
ax2.set_title("Time step 2", usetex=True)
ax3.set_title("Time step 3", usetex=True)
ax4.set_title("Time step 4", usetex=True)
ax0d.imshow(dc0)
ax1d.imshow(dc1)
ax2d.imshow(dc2)
ax3d.imshow(dc3)
ax4d.imshow(dc4)
ax0d.set_title("—", usetex=True)
ax1d.set_title("$t_1 - t_0$", usetex=True)
ax2d.set_title("$t_2 - t_1$", usetex=True)
ax3d.set_title("$t_3 - t_2$", usetex=True)
ax4d.set_title("$t_4 - t_3$", usetex=True)
ax0s.imshow(sc0)
ax1s.imshow(sc1)
ax2s.imshow(sc2)
ax3s.imshow(sc3)
ax4s.imshow(sc4)
ax0s.set_title("—", usetex=True)
ax1s.set_title("$t_1$ scanlines", usetex=True)
ax2s.set_title("$t_2$ scanlines", usetex=True)
ax3s.set_title("$t_3$ scanlines", usetex=True)
ax4s.set_title("$t_4$ scanlines", usetex=True)
ax0b.imshow(su0)
ax1b.imshow(su1)
ax2b.imshow(su2)
ax3b.imshow(su3)
ax4b.imshow(su4)
ax0b.set_title("—", usetex=True)
ax1b.set_title("$t_1$ scanlines (unclipped)", usetex=True)
ax2b.set_title("$t_2$ scanlines (unclipped)", usetex=True)
ax3b.set_title("$t_3$ scanlines (unclipped)", usetex=True)
ax4b.set_title("$t_4$ scanlines (unclipped)", usetex=True)

fig.tight_layout()
fig.set_size_inches(20, 14)
if SAVING:
    fig.savefig("scanlines_plots.png", bbox_inches="tight")
else:
    fig.show()
