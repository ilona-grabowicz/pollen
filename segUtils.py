#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from scipy import ndimage as ndi
from scipy.spatial import distance_matrix

from skimage import filters, transform, util
from skimage.morphology import watershed
from skimage.feature import peak_local_max


# removes points which are considered duplicates
def remove_close_duplicates(points, max_distance=100):
    # calculate distance matrix
    distances = distance_matrix(np.transpose(points), np.transpose(points))
    # mask to avoid deleting both duplicates
    # make diagonal and triangle alays bigger than max deletion distance
    mask = (np.tril(np.ones(np.shape(distances)))*(max_distance+1))
    selection_matrix = np.maximum(distances, mask)
    duplicates =(np.where(selection_matrix < max_distance)[1])
    return( np.delete(points, duplicates, 1) )

# crops square from image wiht specified center and size
def crop_center(img, center,size):
    x,y = center
    top = int(y-(size/2.0))
    bottom = int(y+(size/2.0))
    left = int(x-(size/2.0))
    right = int(x+(size/2.0))
    return img[top:bottom,left:right]

# Debug function - show images from the list
def show_images(images):
    for image in images:
        plt.imshow(image)
        plt.show()

# finds and extracts dark grains using watershead segmentation
def extractDarkGrains(img, size=600, verbose=False):
    img_copy = img
    if verbose:
        plt.imshow(img_copy)
        plt.show()
    # Add margin to be able to take image of size even if grain is at the very edge of picture
    margin_size = int(1+(size/2))
    padding = ((margin_size,margin_size),(margin_size,margin_size),(0,0))
    img_copy = util.pad(img_copy, padding, constant_values=np.mean(img_copy))
    if verbose:
        plt.imshow(img_copy)
        plt.show()
    mask = img_copy

    # make it monochrome (green channel works the best in the case)
    mask = mask[:,:,1]
    if verbose:
        plt.imshow(mask)
        plt.show()

    # blurr to avoid holes in binarization
    blurred = filters.gaussian(mask, sigma=50)
    if verbose:
        plt.imshow(blurred)
        plt.show()

    #binarize - cutoff level relative to mean value to remove those pesky local maximums
    mask =  blurred<(np.mean(blurred)*0.74)
    if verbose:
        plt.imshow(mask)
        plt.show()
    
    #Watershed
    distance = ndi.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, min_distance=1000,footprint=np.ones((50, 50)),
                                labels=mask)

    markers = ndi.label(local_maxi)[0]
    labels = watershed(blurred,markers=markers, mask=mask, compactness=0.5)

    if verbose:
        plt.imshow(distance)
        plt.show()
        plt.imshow(labels)
        plt.show()
    un_labs = np.unique(labels)
    if verbose:
        print(len(np.unique(labels)))
    sizes=[]
    for label in un_labs[1:]:
        sizes.append(np.sum(labels == label))

    # remove results smaller than arbitrary value (1/5 of average pollen size)
    relevant_indices = (np.where(sizes>np.float64(10000))[0])
    maxi_centers = np.where(local_maxi)
    relevant_centers = maxi_centers[0][relevant_indices],maxi_centers[1][relevant_indices],

    relevant_centers = remove_close_duplicates(relevant_centers)

    images = []
    
    # crop images 
    for i in range(0,len(relevant_centers[1])): 
        img_frame =crop_center(img_copy,(relevant_centers[1][i], relevant_centers[0][i]),size)
        if verbose:
            print( np.mean(np.abs(filters.rank.gradient(img_frame[:,:,1], np.ones((5,5))))))
            plt.imshow(img_frame)
            plt.show()
        images.append(img_frame)
        
    return(images)
