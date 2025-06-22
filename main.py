"""
A library to compare images and highlight the differences.
In particular it can be used in tests where the output of a
function is an image.
"""

import numpy as np
from PIL import Image

# Compare two images pixel by pixel
img_1 = np.array(Image.open("man_with_boxes_1.jpg"))
img_2 = np.array(Image.open("man_with_boxes_2.jpg"))

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

# Check the image directly, find the diffrences and
## hightlight them in a side by side comparison

# Include pre check of size, try resizing both to the other size
## PIL library probably best here

# Image may be cropped, move the smaller image around the
## larger one to find a fit (use the OVImage?)

# Compare metadata
# Use the os library and perhaps json?

# Include suggestions for the difference and how to correct for it

# Include a threshold margin of error
