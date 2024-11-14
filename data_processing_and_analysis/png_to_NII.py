
'''
Recives: a path towards a directory that contains all slices, as .png
a path towards a dir with the original .nii file, for corectly re-constructing the header
Returns: the .nii file

Important: 1) trebuie sa existe toate .png-utrile. Catre un png per slice. Altfel nu mereg
           2) Trebuie ca ultimul nr dupa _ din denumirea .png sa fie nr slice-ului 
'''

import os
import shutil
import nibabel as nib
import numpy as np
import cv2


#PNG_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_traduced_to_PNG_from_pas1'
PNG_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\tst_out'
ORIGINAL_NII_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_fara_gz\BraTS2021_00002\BraTS2021_00002_seg.nii'

OUTPUT_RECONSTRUCTION_FILE = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Reconstructed'


# load
scan_loadfull = nib.load(ORIGINAL_NII_PATH, mmap=False)
scan = scan_loadfull.get_fdata()
#scan_path = ORIGINAL_NII_PATH
scan_affine = scan_loadfull.affine
scan_header = scan_loadfull.header

#print(scan_loadfull)


# acum se suprasciru valorile 
for filename in os.listdir(PNG_PATH):
    #print(filename)
    # ge t the index of the slicve
    index_slice = int(filename.split('_')[-1])
    #print(index_slice)

    slice_png = cv2.imread(PNG_PATH + '\\' + filename + '\\' + 'seg' + '.png')

    scan[:,:,index_slice] = slice_png[:, :, 0] # grayscaleul are acelasi nivel de gri pe toate canalelel


new_scan = nib.Nifti1Image(scan, scan_affine, scan_header)

nib.save(new_scan, OUTPUT_RECONSTRUCTION_FILE + '\\' + 'file')
#nib.save(scan_loadfull, OUTPUT_RECONSTRUCTION_FILE + '\\' + 'file') # ASA SE SALVEAZA CORECT pt vizualizre pe brainviewer. Ce faceam eu la "Pas2 si 3" strica headerul


