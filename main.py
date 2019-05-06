import numpy as np
from settings import Settings
from utils import *
from preprocess import data_processor

lungPatients, labels = load_data()

size = Settings.size
no_slices = Settings.no_slices

imageData = []
for num, patient in enumerate(lungPatients):
    if num % 100 == 0:
        print('Saved -', num)
    try:
        img_data, label = data_processor(patient, labels, size=size, noslices=no_slices)
        imageData.append([img_data, label, patient])
        # imageData.append([aug_data, label, patient+'_aug'])
    except KeyError as e:
        print('Data is unlabeled')

        
##Results are saved as numpy file
np.save('imageDataNew-{}-{}-{}.npy'.format(size, size, no_slices), imageData)