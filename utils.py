import os
import pydicom
import math
import pandas as pd
# import matplotlib.pyplot as plt
from settings import Settings

def mean(l):
    return sum(l) / len(l)

def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        if (count < Settings.no_slices):
            yield l[i:i + n]
            count = count + 1

def get_label(patient, labels_df):
    return(labels_df.at[patient, 'cancer'])

def load_data(data_directory=Settings.data_dir, labels_file=Settings.labels_file):
    lung_patients = os.listdir(data_directory)
    labels_df = pd.read_csv(labels_file, index_col=0)
    return(lung_patients, labels_df)

def load_patient(labels_df, patient, data_directory=Settings.data_dir):
    label = labels_df.at[patient, 'cancer']
    
    path = data_directory + patient
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    
    return(label, slices)

# def display_mid_slice(slices):
#     ind = math.floor(len(slices)/2)
#     plt.imshow(slices[ind], interpolation='nearest')
#     plt.show()
