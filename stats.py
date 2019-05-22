import numpy as np

imageData = np.load('imageDataNew-64-64-20.npy')

cancer = 0
benign = 0

for each in imageData:
    if np.array_equal(each[1], np.array([0, 1])):
        cancer += 1
    elif (np.array_equal(each[1], np.array([1, 0]))):
        benign += 1

print(imageData.shape)
print('{} patients are diagnosed with cancer\n{} patiens do not have cancer'.format(cancer, benign))

augData = np.load('augmentedDataNew-64-64-20.npy')

a_cancer = 0

for each in augData:
    if np.array_equal(each[1], np.array([0, 1])):
        a_cancer += 1

print('{} augmented patiens have cancer'.format(a_cancer))
print(augData.shape)

[print(pat[2]) for pat in augData[1:5]]

np.random.shuffle(augData)

[print(pat[2]) for pat in augData[1:5]]
