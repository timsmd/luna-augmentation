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
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.1.
        # But we only blur about 10% of all images.
        iaa.Sometimes(0.1,
            iaa.GaussianBlur(sigma=(0, 0.1))
        ),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (1, 1.2), "y": (1, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            shear=(-2, 2)
        )
    ], random_order=True)

    aug_slices = seq.augment_images(aug_slices)
    
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