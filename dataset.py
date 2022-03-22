import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import csv
import os

# %% codecell
# general parameters
run = 'example'
version = 1
slice_size = 128
step_size = 32

# general functions
def binarize(slice):
    grey = copy.deepcopy(slice)
    grey[grey > 150] = 150
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    _, bin = cv2.threshold(grey, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bin

def pad(image):
    pad = 100
    min = np.min(image[0, :])
    padded = np.pad(image, [(pad, pad), (pad, pad)], mode = 'constant')
    padded[padded == 0] = min

    return padded

# %% codecell
# load and process images
GDS_images = []
SEM_images = []
SEMb_images = []
GDS_dir = 'data/GDS/' + run + '/'
SEM_dir = 'data/SEM/' + run + '/'
for SEM_path in os.listdir(SEM_dir):
    SEM_image = cv2.imread(SEM_dir + SEM_path, 0)
    if SEM_image is not None:
        pattern_size = SEM_image.shape[::-1]
        SEMb_image = binarize(SEM_image)
        SEM_image = pad(SEM_image)
        SEMb_image = pad(SEMb_image)
        SEM_images.append(SEM_image)
        SEMb_images.append(SEMb_image)
for GDS_path in os.listdir(GDS_dir):
    GDS_image = cv2.imread(GDS_dir + GDS_path, 0)
    if GDS_image is not None:
        GDS_image = cv2.resize(GDS_image, pattern_size)
        GDS_image = pad(GDS_image)
        GDS_images.append(GDS_image)
image_size = GDS_images[0].shape

# %% codecell
# cut slices and save
xs = ys = 0
xe = ye = slice_size
im = 0
rows = []
dataset_dir = 'datasets/' + str(run) + '_v' + str(version)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
for j in range(len(SEM_images)):
    print('Slicing and saving Image ' + str(j + 1) + '/'
        + str(len(GDS_images)) +  '...')
    xs = ys = 0
    xe = ye = slice_size
    while xe < image_size[1]:
        ys = 0
        ye = slice_size
        while ye < image_size[0]:
            if (GDS_images[j][ys:ye, xs:xe].shape == (slice_size, slice_size)):
                cv2.imwrite(dataset_dir + '/GDS_' + str(im) + '.jpg',
                    GDS_images[j][ys:ye, xs:xe])
                cv2.imwrite(dataset_dir + '/SEM_' + str(im) + '.jpg',
                    SEM_images[j][ys:ye, xs:xe])
                cv2.imwrite(dataset_dir + '/SEMb_' + str(im) + '.jpg',
                    SEMb_images[j][ys:ye, xs:xe])
                rows.append(['GDS_' + str(im) + '.jpg',
                    'SEM_' + str(im) + '.jpg', 'SEMb_' + str(im) + '.jpg'])
                im += 1
            ye += step_size
            ys += step_size
        xe += step_size
        xs += step_size

# create .csv of image filenames
csvpath = 'datasets/' + str(run) + '_v' + str(version) + ".csv"

with open(csvpath, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['GDS', 'SEM', 'SEMb'])
    csvwriter.writerows(rows)
