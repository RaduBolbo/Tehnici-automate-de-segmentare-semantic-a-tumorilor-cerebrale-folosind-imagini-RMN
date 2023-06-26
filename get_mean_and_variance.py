'''
This script gets the mean and variance of each slice in all the PNG plices in the dataset

This will be performed juist on the reduced trainign dataset, as we can inffer it is the same for the whole distribution

'''
import os
import numpy as np
import cv2 as cv


PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'

#arr.reshape(-1)

# sumofmeans = the mean pover a siungle example
sumofmeans_list_t1 = []
sumofmeans_list_t1ce = []
sumofmeans_list_t2 = []
sumofmeans_list_flair = []

psum_t1 = 0
psum_t1ce = 0
psum_t2 = 0
psum_flair = 0

psumsq_t1_list = []
psumsq_t1ce_list = []
psumsq_t2_list = []
psumsq_flair_list = []

no_examples = 0

for dir_name in os.listdir(PATH):

    EXAPLE_PATH = PATH + '\\' + dir_name

    for file_name in os.listdir(EXAPLE_PATH):
        FILE_PATH = EXAPLE_PATH + '\\' + file_name
        img = cv.imread(FILE_PATH, cv.IMREAD_GRAYSCALE)/255
        flattened_img = img.reshape(-1)
        if file_name == 't1.png':
            sumofmeans_list_t1.append(np.sum(flattened_img))
            psumsq_t1_list.append(np.sum(flattened_img**2))
        elif file_name == 't1ce.png':
            sumofmeans_list_t1ce.append(np.sum(flattened_img))
            psumsq_t1ce_list.append(np.sum(flattened_img**2))
        elif file_name == 't2.png':
            sumofmeans_list_t2.append(np.sum(flattened_img))
            psumsq_t2_list.append(np.sum(flattened_img**2))
        elif file_name == 'flair.png':
            sumofmeans_list_flair.append(np.sum(flattened_img))
            psumsq_flair_list.append(np.sum(flattened_img**2))
    
    no_examples += 1

count = no_examples * 240 * 240

list_t1 = []
list_t1ce = []
list_t2 = []
list_flair = []

for x in sumofmeans_list_t1:
    list_t1.append(x/count)

for x in sumofmeans_list_t1ce:
    list_t1ce.append(x/count)

for x in sumofmeans_list_t2:
    list_t2.append(x/count)

for x in sumofmeans_list_flair:
    list_flair.append(x/count)

mean_t1 = sum(list_t1)
mean_t1ce = sum(list_t1ce)
mean_t2 = sum(list_t2)
mean_flair = sum(list_flair)

list_t1 = []
list_t1ce = []
list_t2 = []
list_flair = []

for x in psumsq_t1_list:
    list_t1.append(x/count)

for x in psumsq_t1ce_list:
    list_t1ce.append(x/count)

for x in psumsq_t2_list:
    list_t2.append(x/count)

for x in psumsq_flair_list:
    list_flair.append(x/count)

#print(psumsq_t1_list[0:4])
#print(list_t1[0:4])
#print(sum(list_t1))
#print(sum(list_t1)-mean_t1**2)

total_std_t1 = np.sqrt(sum(list_t1)-mean_t1**2)
total_std_t1ce = np.sqrt(sum(list_t1ce)-mean_t1ce**2)
total_std_t2 = np.sqrt(sum(list_t2) -mean_t2**2)
total_std_flair = np.sqrt(sum(list_flair)-mean_flair**2)
    
#print(mean_t1)
#print(mean_t1ce)
#print(mean_t2)
#print(mean_flair)

#print(total_std_t1)
#print(total_std_t1ce)
#print(total_std_t2)
#print(total_std_flair)



'''
PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'

#arr.reshape(-1)

# sumofmeans = the mean pover a siungle example
sumofmeans_t1 = 0
sumofmeans_t1ce = 0
sumofmeans_t2 = 0
sumofmeans_flair = 0

sumofstd_t1 = 0
sumofstd_t1ce = 0
sumofstd_t2 = 0
sumofstd_flair = 0

no_examples = 0

for dir_name in os.listdir(PATH):

    EXAPLE_PATH = PATH + '\\' + dir_name

    for file_name in os.listdir(EXAPLE_PATH):
        FILE_PATH = EXAPLE_PATH + '\\' + file_name
        img = cv.imread(FILE_PATH, cv.IMREAD_GRAYSCALE)
        flattened_img = img.reshape(-1)
        if file_name == 't1.png':
            sumofmeans_t1 += np.mean(flattened_img)
            sumofstd_t1 += np.std(flattened_img)
        elif file_name == 't1ce.png':
            sumofmeans_t1ce += np.mean(flattened_img)
            sumofstd_t1ce += np.std(flattened_img)
        elif file_name == 't2.png':
            sumofmeans_t2 += np.mean(flattened_img)
            sumofstd_t2 += np.std(flattened_img)
        elif file_name == 'flair.png':
            sumofmeans_flair += np.mean(flattened_img)
            sumofstd_flair += np.std(flattened_img)
    
    no_examples += 1

mean_t1 = sumofmeans_t1/no_examples
mean_t1ce = sumofmeans_t1ce/no_examples
mean_t2 = sumofmeans_t2/no_examples
mean_flair = sumofmeans_flair/no_examples

std_t1 = sumofstd_t1/no_examples
std_t1ce = sumofstd_t1ce/no_examples
std_t2 = sumofstd_t2/no_examples
std_flair = sumofstd_flair/no_examples
    
#print(mean_t1)
#print(mean_t1ce)
#print(mean_t2)
#print(mean_flair)

#print(std_t1)
#print(std_t1ce)
#print(std_t2)
#print(std_flair)

'''









