import numpy as np
import math
import cv2
from settings import Settings
from utils import *

def data_processor(patient, labels_df, size=Settings.size, noslices=Settings.no_slices, visualize=False):
    label, slices = load_patient(labels_df=labels_df, patient=patient)

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])

    slices = [cv2.resize(np.array(each_slice.pixel_array), (size, size)) for each_slice in slices]

    if visualize:
        display_mid_slice(slices)

    new_slices = []
    chunk_sizes = math.floor(len(slices) / noslices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    
    patient_data = {
        'img_data': np.array(new_slices),
        'patient': patient,
        'label': label
    }

    return(patient_data)