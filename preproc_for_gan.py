import numpy as np
import math
import cv2
from settings import Settings
from utils import *

size = Settings.size
no_slices = Settings.no_slices

def dicom_to_array(patient, labels_df, size=Settings.size):
    label, slices = load_patient(labels_df=labels_df, patient=patient)
    resized = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]
    
    return_slices = []
    for each in resized:
        return_slices.append(np.array(each)[np.newaxis, :, :])
    return label, return_slices

imageData = np.load('imageDataNew-64-64-20.npy')

cancerImages = []

for patient in imageData:
    if np.array_equal(patient[1], [0, 1]):
        cancerImages.append(patient[0])

cancerImages = np.array(cancerImages)

print(cancerImages.shape)
        
##Results are saved as numpy file
np.save('CancerImages-{}-{}-{}.npy'.format(size, size, no_slices), cancerImages)
print('finished...')