import numpy as np
import math
import cv2
from settings import Settings
from utils import *

def dicom_to_array(patient, labels_df, size=Settings.size):
    label, slices = load_patient(labels_df=labels_df, patient=patient)
    resized = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]
    
    return_slices = []
    for each in resized:
        return_slices.append(np.array(each)[np.newaxis, :, :])
    return(return_slices)


lungPatients, labels = load_data()

size = Settings.size

imageData = []
for num, patient in enumerate(lungPatients):
    if num % 100 == 0:
        print('Saved -', num)
    try:
        patient_data = []
        patient_data = dicom_to_array(patient=patient, labels_df=labels, size=size)
        for each in patient_data:
            imageData.append(np.array(each))
    except KeyError as e:
        print('Data is unlabeled')
    except Exception as er:
        print(er)

        
##Results are saved as numpy file
np.save('imageDataNew-{}-{}.npy'.format(size, size), np.array(imageData))