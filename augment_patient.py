import numpy as np
import math
import cv2
from imgaug import augmenters as iaa
from settings import Settings
from utils import *

def augment_patient(patient, labels_df, iteration=0, size=Settings.size, noslices=Settings.no_slices, visualize=False):
    label, slices = load_patient(labels_df=labels_df, patient=patient)

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    aug_slices = np.array([np.array(each_slice.pixel_array) for each_slice in slices])

    # Augmentation formula
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5), # vertical flips
        iaa.ContrastNormalization((0.90, 1.1)),
        iaa.Affine(
            scale={"x": (0.9, 1.2), "y": (0.9, 1.2)},
            rotate=(-90, 90),
            shear=(-10, 10)
        )
    ], random_order=True)

    images = []

    aug_slices = np.transpose(aug_slices, (1, 2, 0))

    images.append(aug_slices)

    images = seq.augment_images(images)

    aug_slices = images[0]

    aug_slices = np.transpose(aug_slices, (2, 0, 1))
    
    resized_aug = []
    for i in range(len(aug_slices)):
        resized_aug.append(cv2.resize(aug_slices[i], (size, size)))
    resized_aug = np.array(resized_aug)

    if visualize:
        display_mid_slice(resized_aug)
 
    new_aug_slices = []
    chunk_sizes = math.floor(len(resized_aug) / noslices)
    for slice_chunk in chunks(resized_aug, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_aug_slices.append(slice_chunk)

    augmented_patient_data = {
        'img_data': np.array(new_aug_slices),
        'patient': 'aug_' + patient + '_t{}'.format(iteration),
        'label': label
    }

    return(augmented_patient_data)