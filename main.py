"""
A library to compare images and highlight the differences.
In particular it can be used in tests where the output of a
function is an image.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def rescale_image(img_1: str | Path, img_2: str | Path):
    # Check the image size, two paths from there - resize, check if cropped
    width_1, height_1 = img_1.size
    width_2, height_2 = img_2.size
    width_ratio = width_2 / width_1
    height_ratio = height_2 / height_1
    if width_ratio != height_ratio:
        print("Image is not evenly scaled, perhaps it is cropped or extended.")
    if width_ratio == 1 and height_ratio != 1:
        print("Only the height has been changed.")
    elif width_ratio != 1 and height_ratio == 1:
        print("Only the width has been changed.")
    return img_2.resize((width_1, height_1))


def check_if_cropped(img_1, img_2):
    img_rgb = cv2.imread("mario.png")
    template = cv2.imread("mario_coin.png")
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1], strict=False):  # Switch columns and rows
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("result.png", img_rgb)


# Compare two images pixel by pixel
img_1 = np.array(Image.open("man_with_boxes_1.jpg"))
img_2 = np.array(Image.open("man_with_boxes_alt.jpg"))

# Check if the shapes are the same and pixels are identical
if img_1.shape == img_2.shape and np.all(img_1 == img_2):
    print("Images are identical!")
else:
    print("Images are different.")

out = np.zeros_like(img_2)

mask = img_1 != img_2
out[mask] = img_2[mask]

out_img = Image.fromarray(out)

out_img.save("differences_1.jpg")

# Check the image directly, find the differences and
## hightlight them in a side by side comparison

# Include pre check of size, try resizing both to the other size
## PIL library probably best here

# Image may be cropped, move the smaller image around the
## larger one to find a fit (use the OVImage?)

# Compare metadata
# Use the os library and perhaps json?

# Include suggestions for the difference and how to correct for it

# Include a threshold margin of error


def main(img_1_loc, img_2_loc):
    # Open the images
    img_1 = Image.open(img_1_loc)
    img_2 = Image.open(img_2_loc)
    # Check if they are identical or not
    # Compare two images pixel by pixel
    img_1_array = np.array(img_1)
    img_2_array = np.array(img_2)

    # Check if the shapes are the same and pixels are identical
    if img_1_array.shape == img_2_array.shape and np.all(
        img_1_array == img_2_array
    ):
        print("Images are identical!")
    else:
        print("Images are different.")

    # If the size is the issue then check if it is cropped

    # If not cropped check if it can rescale

    # If not find the differences

    # Categorise the difference if possible, e.g. text, noise, compressed

    # Output composite image with img_1, img_2, img_diff,
    ## and stats and test results


# TODO: Include test for image compression
# TODO: Test vector images
# TODO: Test for text changes
# TODO: Create standard output showing difference
