# Required Libraries
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from pathlib import Path
import argparse
import numpy


# Argument parsing variable declared
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",required=True,help="Path to folder")
ap.add_argument("-e", "--mask",required=True,help="Path to folder")
args = vars(ap.parse_args())

# Find all the images in the provided images folder

mypath1 = args["image"]
mypath2 = args["mask"]
onlyfiles1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
images = numpy.empty(len(onlyfiles1), dtype=object)
masks = numpy.empty(len(onlyfiles2), dtype=object)

# Iterate through every image
# and resize all the images.
for n in range(0, len(onlyfiles1)):
    path1 = join(mypath1, onlyfiles1[n])
    path2 = join(mypath2, onlyfiles2[n])
    images[n] = cv2.imread(join(mypath1, onlyfiles1[n]),cv2.IMREAD_UNCHANGED)
    masks[n] = cv2.imread(join(mypath2, onlyfiles2[n]),cv2.IMREAD_UNCHANGED)
    
    # Load the image in img variable
    img = cv2.imread(path1, 1)
    msk= cv2.imread(path2, 1)
    resize_width = int(256)
    resize_hieght = int(256)
    resized_dimensions = (resize_width, resize_hieght)
    resized_msk = cv2.resize(msk, resized_dimensions, interpolation=cv2.INTER_AREA)

    # Define a resizing Scale
    # To declare how much to resize
    mask_img = cv2.bitwise_or(resized_msk, img)
    # Create resized image using the calculated dimensions
    # Save the image in Output Folder
    cv2.imwrite('output/' + str(n) + '_resized.png', mask_img)

    
print("Images masked Successfully")
