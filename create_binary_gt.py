
'''
INTREBARI: 1) cu affine - in primul rand ce sens are AFFINE pt un SINGUR CANAL?
            2) E mai bine de facut partea a 2-a sa primeasca la Input
            strict tumoareasegmentrata, sau sa aiba ca un "ATTENTION"
            pe ce a segmentat precedenta
            3) DE CE NU SUNT NR COMPLEXE?
            4) Nu ma prind daca T1CE este doar T1 cu contrast crescut artificial (software) si daca mai are rost pastarrea T1 original

The script enters all te examples in the given folder and creates
an additional .nii file with a binary GT.
'''
import os
import shutil
import nibabel as nib
import numpy as np

# Reduced Dtataset
#DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Reduced_Dataset\Training_Dataset'
#DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Test_Dataset'
#DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'

DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_fara_gz'

######################
# 1) se dezarhiveaza .gz-urile
######################

no_of_deleted_slices = 0

for dir_name_2 in os.listdir(DIR_PATH):
    DIR_PATH_2 = DIR_PATH + '\\' + dir_name_2

    #print(DIR_PATH_2)

    for file_name in os.listdir(DIR_PATH_2):
        FILE_PATH = DIR_PATH_2 + '\\' + file_name
        ##print(FILE_PATH)
        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg': # in cazul in care este fisier seg
            scan_seg = nib.load(FILE_PATH).get_fdata()
            scan_seg_path = FILE_PATH
            scan_seg_affine = nib.load(FILE_PATH).affine
            # create new path
            scan_boolseg_path = FILE_PATH.split('.')[0][0:-3] + 'boolseg.nii'

    # building the boolean gt
    scan_boolseg_modified = (scan_seg >= 1) * 1

    # ndarray -> nii transformatpion:
    scan_boolseg_modified = nib.Nifti1Image(scan_boolseg_modified, scan_seg_affine)

    # saving the neww cleaned nii
    nib.save(scan_boolseg_modified, scan_boolseg_path)
