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
    print('processing patient # {} - {} out of {}'.format(num, patient, len(lungPatients)))

    try:
        if get_label(patient, labels) == 1:
            print('cancer - {}'.format(num))
            augmented_data = []
            for i in range(Settings.aug_counter):
                augmented_data.append(augment_patient(patient=patient, labels_df=labels, iteration=i, size=size, noslices=no_slices, visualize=False))
            for each in augmented_data:
                augData.append([each['img_data'], each['label'], each['patient']])
    except KeyError as e:
        print('Data is unlabeled')
    except Exception as er:
        print(er)
        
np.save('augmentedDataNew-{}-{}-{}.npy'.format(size, size, no_slices), augData)