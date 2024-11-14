
'''
The script enters all te examples in the given folder and creates
computes what is the tumor percentage (meam percentage) - for each slice, 3D scan and dataset.
'''
import os
import shutil
import nibabel as nib
import numpy as np
from collections import Counter

DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Experiment'

######################
# 1) se dezarhiveaza .gz-urile
######################

# declaring a list which will stre all 3D scan lists
triD_scan_list = []

for dir_name_2 in os.listdir(DIR_PATH):
    DIR_PATH_2 = DIR_PATH + '\\' + dir_name_2

    ##print(DIR_PATH_2)

    for file_name in os.listdir(DIR_PATH_2):
        FILE_PATH = DIR_PATH_2 + '\\' + file_name
        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] == 'boolseg': # in cazul in care este fisier seg
            ##print(FILE_PATH)
            scan_seg = nib.load(FILE_PATH).get_fdata()
            scan_seg_path = FILE_PATH

            # declaring a list which will stre all slices lists
            slices_list = []
            for i in range(scan_seg.shape[2]):
                # compute histogram of scan_t1
                unique_scan_t1, counts_scan_t1 = np.unique(scan_seg[:, :, i], return_counts=True)
                d = dict(zip(unique_scan_t1, counts_scan_t1))
                slices_list.append(d)
            d_merged = {}
            for d in slices_list:
                a_counter = Counter(d_merged)
                b_counter = Counter(d)
                add_dict = a_counter + b_counter
                d_merged = dict(add_dict)

            triD_scan_list.append(d_merged)


#print(triD_scan_list)

# now, triD_scan_list contains one dict for each scan in the given folder
# Compute fractions for each dict:

triD_scanfractions_list = []

for d in triD_scan_list:
    no_zeros = d.get(0.0, 0)
    no_ones = d.get(1.0, 0)
    fraction_zeros = no_zeros / (no_zeros + no_ones)
    fraction_ones = no_ones / (no_zeros + no_ones)
    triD_scanfractions_list.append([fraction_zeros, fraction_ones])

#print('Pentru fiecare scan, procentajele fraction_zeros-fraction_ones sunt:')
#print(triD_scanfractions_list)

sum_by_axes = np.sum(triD_scanfractions_list, axis = 0)
fraction_zeros = sum_by_axes[0] / len(triD_scanfractions_list)
fraction_ones = sum_by_axes[1] / len(triD_scanfractions_list)
#print('Ca o medie: ')
#print('fraction_zeros: ' + str(fraction_zeros) + ' fraction_ones: ' + str(fraction_ones))












