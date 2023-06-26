
'''
The script enters all te examples in the given folder and deletes the blank
slices which contain no scaned brain from the .nii file

VARIANTA 1 - salveaza fara probleme la precizia zecimalelor, dar uneori se blpocheaza
VARIANTA 2- probleme la precizia zecimaleor

Prerequisites:
    - the script DOES NOT PROVIDE support for BOOLSEG data !
'''



import os
import shutil
import nibabel as nib
import numpy as np

# Reduced dataset
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
        if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
            scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1 = scan_t1_loadfull.get_fdata()
            scan_t1_path = FILE_PATH
            scan_t1_affine = scan_t1_loadfull.affine
            ##print(scan_t1_affine)
        elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
            scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t2 = scan_t2_loadfull.get_fdata()
            scan_t2_path = FILE_PATH
            scan_t2_affine = scan_t2_loadfull.affine
        elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
            scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1ce = scan_t1ce_loadfull.get_fdata()
            scan_t1ce_path = FILE_PATH
            scan_t1ce_affine = scan_t1ce_loadfull.affine
        elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair': # in cazul in care este fisier flair
            scan_flair_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_flair = scan_flair_loadfull.get_fdata()
            scan_flair_path = FILE_PATH
            scan_flair_affine = scan_flair_loadfull.affine
        elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg': # in cazul in care este fisier seg
            scan_seg_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_seg = scan_seg_loadfull.get_fdata()
            scan_seg_path = FILE_PATH
            scan_seg_affine = scan_seg_loadfull.affine

    # acum ca toate fisierele su fost incarcate, se incepe verificarea

    # iterate through t1 slices:
    list_of_index = []
    for i in range(scan_t1.shape[2]):
        # compute histogram of scan_t1
        unique_scan_t1, counts_scan_t1 = np.unique(scan_t1[:, :, i], return_counts=True)
        d = dict(zip(unique_scan_t1, counts_scan_t1))
        if len(d) == 1: # when the slice has one single value (this happens only when blank)
            # all slices with i idex will be deleted. Indexes are stored in a list
            no_of_deleted_slices += 1
            list_of_index.append(i)

    # deleting slices
    scan_t1_modified = np.delete(scan_t1, obj = list_of_index, axis = 2)
    scan_t1ce_modified = np.delete(scan_t1ce, obj=list_of_index, axis=2)
    scan_t2_modified = np.delete(scan_t2, obj=list_of_index, axis=2)
    scan_flair_modified = np.delete(scan_flair, obj=list_of_index, axis=2)
    scan_seg_modified = np.delete(scan_seg, obj=list_of_index, axis=2)

    ################################ VARIANTA 1 ##########################################

    scan_t1_modified = nib.Nifti1Image(scan_t1_modified, scan_t1_affine)
    scan_t1ce_modified = nib.Nifti1Image(scan_t1ce_modified, scan_t1ce_affine)
    scan_t2_modified = nib.Nifti1Image(scan_t2_modified, scan_t2_affine)
    scan_flair_modified = nib.Nifti1Image(scan_flair_modified, scan_flair_affine)
    scan_seg_modified = nib.Nifti1Image(scan_seg_modified, scan_seg_affine)

    nib.save(scan_t1_modified, scan_t1_path)
    nib.save(scan_t1ce_modified, scan_t1ce_path)
    nib.save(scan_t2_modified, scan_t2_path)
    nib.save(scan_flair_modified, scan_flair_path)
    nib.save(scan_seg_modified, scan_seg_path)

    ################################### VARIANTA 1 #######################################

    ################################### VARIANTA 2 #######################################
    # VARIANTA2 se ruleaza in cazul in care VARIANTA 1 nu mai merge, in mod total neasteptat
    '''
    # ndarray -> nii transformatpion:
    scan_t1_modified = scan_t1_loadfull.__class__(scan_t1_modified, scan_t1_affine, scan_t1_loadfull.header)
    scan_t1ce_modified = scan_t1ce_loadfull.__class__(scan_t1ce_modified, scan_t1ce_affine, scan_t1ce_loadfull.header)
    scan_t2_modified = scan_t2_loadfull.__class__(scan_t2_modified, scan_t2_affine, scan_t2_loadfull.header)
    scan_flair_modified = scan_flair_loadfull.__class__(scan_flair_modified, scan_flair_affine,scan_flair_loadfull.header)
    scan_seg_modified = scan_seg_loadfull.__class__(scan_seg_modified, scan_seg_affine, scan_seg_loadfull.header)

    # saving the neww cleaned niis with the name of old ones
    #print(scan_t1_path)
    os.remove(scan_t1_path)
    nib.save(scan_t1_modified, scan_t1_path)
    os.remove(scan_t1ce_path)
    nib.save(scan_t1ce_modified, scan_t1ce_path)
    os.remove(scan_t2_path)
    nib.save(scan_t2_modified, scan_t2_path)
    os.remove(scan_flair_path)
    nib.save(scan_flair_modified, scan_flair_path)
    os.remove(scan_seg_path)
    nib.save(scan_seg_modified, scan_seg_path)
    '''
    ################################### VARIANTA 2 #######################################



#print('blank slaice clean complete.')
#print('deleted ' + str(no_of_deleted_slices) + ' blank slices')














