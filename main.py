"""
A library to compare images and highlight the differences.
In particular it can be used in tests where the output of a
function is an image.
"""

import cv2
import numpy as np
from PIL import Image


def rescale_image(img_1: Image, img_2: Image):
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
    else:
        print(f"Image has been rescaled by a factor of: {width_ratio}")
    return img_2.resize((width_1, height_1))


def check_if_cropped(img_1: np.array, img_2: np.array):
    # Find which image is smaller
    if img_1.shape[0] <= img_2.shape[0] and img_1.shape[1] <= img_2.shape[1]:
        img_cropped = img_1
        img_full = img_2
    elif img_1.shape[0] >= img_2.shape[0] and img_1.shape[1] >= img_2.shape[1]:
        img_cropped = img_2
        img_full = img_1
    else:
        print("Image seems to be reshaped or distorted.")
        return None

    w, h = img_cropped.shape[:-1]

    res = cv2.matchTemplate(img_full, img_cropped, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1], strict=False):  # Switch columns and rows
        cv2.rectangle(img_full, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("highlight_crop.png", img_full)

    if len(loc) > 0:
        return True
    else:
        return False

    # img_rgb = cv2.imread("mario.png")
    # template = cv2.imread("mario_coin.png")
    # w, h = template.shape[:-1]

    # res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1], strict=False):  # Switch columns and rows
    #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # cv2.imwrite("result.png", img_rgb)


def find_difference(img_1: np.array, img_2: np.array):
    # Making a blank array in the shape of img_2
    out = np.zeros_like(img_2)
    # Finding where the difference is
    mask = img_1 != img_2
    out[mask] = img_2[mask]
    # Converting back to image, save & return
    out_img = Image.fromarray(out)
    out_img.save("differences_1.jpg")  # TODO add naming convention
    return out_img


# Compare two images pixel by pixel
img_1 = np.array(Image.open("man_with_boxes_1.jpg"))
img_2 = np.array(Image.open("man_with_boxes_alt.jpg"))

# Check if the shapes are the same and pixels are identical
if img_1.shape == img_2.shape and np.all(img_1 == img_2):
    print("Images are identical!")
else:
    print("Images are different.")
print(img_1.shape)
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
        return None
    elif img_1_array.shape == img_2_array.shape and np.all(
        img_1_array != img_2_array
    ):
        print("Difference has been detected. Identifying changes...")
        img_diff = find_difference(img_1, img_2)
        return img_diff
    else:
        print("""Image has a different shape.
              Checking for cropping or reshaping...""")
    # If the size is the issue then check if it is cropped
    if check_if_cropped(img_1_array, img_2_array):
        return None  # check logic here
    else:
        # If not cropped check if it can rescale
        rescaled_img = rescale_image(img_1, img_2)
        rescaled_img.save("Rescaled_result.jpg")
    # Look at combo changes, e.g. edit and rescale
    # Drop threshold in crop check, then apply the diff check to match
    # If not find the differences

    # Categorise the difference if possible, e.g. text, noise, compressed

    # Output composite image with img_1, img_2, img_diff,
    ## and stats and test results


# TODO: Include test for image compression
# TODO: Test vector images
# TODO: Test for text changes
# TODO: Create standard output showing difference
