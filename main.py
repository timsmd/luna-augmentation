import numpy as np
from settings import Settings
from utils import *
from data_processor import data_processor
from augment_patient import augment_patient

lungPatients, labels = load_data()

size = Settings.size
no_slices = Settings.no_slices

imageData = []
augData = []

for num, patient in enumerate(lungPatients):
    if num % 100 == 0:
        print('Saved -', num)
    try:
        patient_data = data_processor(patient=patient, labels_df=labels, size=size, noslices=no_slices, visualize=False)
        print('processing patient # {} - {} out of {}'.format(num, patient, len(lungPatients)))
        if get_label(patient, labels) == 1:
            print('cancer - {}'.format(num))
            augmented_data = []
            for i in range(Settings.aug_counter):
                augmented_data.append(augment_patient(patient=patient, labels_df=labels, iteration=i, size=size, noslices=no_slices, visualize=False))
            for each in augmented_data:
                augData.append([each['img_data'], each['label'], each['patient']])
    
        imageData.append([patient_data['img_data'], patient_data['label'], patient_data['patient']])
    except KeyError as e:
        print('Data is unlabeled')
    except Exception as er:
        print(er)

        
##Results are saved as numpy file
np.save('imageDataNew-{}-{}-{}.npy'.format(size, size, no_slices), imageData)
np.save('augmentedDataNew-{}-{}-{}.npy'.format(size, size, no_slices), augData)
print(len(imageData))
print(len(augData))