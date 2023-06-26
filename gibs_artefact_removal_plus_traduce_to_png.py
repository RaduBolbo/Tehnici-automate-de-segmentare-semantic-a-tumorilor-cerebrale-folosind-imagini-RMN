
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import SimpleITK as sitk
from skimage import exposure
import skimage
import os
from PIL import Image

from gibbs_artifact_removal_by_RafaelNH import gibbs_removal

#Train
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'

#Val
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Val_Dataset'

# Official_Validation_dataset
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\gibbsartremoved_PNG_Val_Official_Dataset'

########### CORECTAT

#Train
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset_CORECT\gibremoved_PNG_CORECT_Train_Dataset'

#Val
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset_CORECT\gibremoved_PNG_CORECT_Val_Dataset'

# Test
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset_CORECT\gibremoved_PNG_CORECT_Test_Dataset'

conversion_no_bits = 8

def gibbs_artifact_removal(scan):

    #corrected_scan = gibbs_removal(scan, slice_axis=2, n_points=3)
    corrected_scan = gibbs_removal(scan, slice_axis=2, n_points=3)

    return corrected_scan

for dir_name_2 in os.listdir(INPUT_PATH):
    INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
    OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2
    '''
    if os.path.isdir(OUTPUT_PATH_2) == False:
        os.mkdir(OUTPUT_PATH_2)
    '''
    #print(dir_name_2)
    for file_name in os.listdir(INPUT_PATH_2):
        FILE_PATH = INPUT_PATH_2 + '\\' + file_name
        OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '_slice_'
        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
            scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1 = scan_t1_loadfull.get_fdata()
            scan_t1_path = FILE_PATH

            # Biass Field removal:
            scan_t1 = gibbs_artifact_removal(scan_t1)
            ##print('b')
            ##print(np.min(scan_t1), np.max(scan_t1))
            ##print(scan_t1.shape)

            # Save as PNG
            for i in range(scan_t1.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t1[:,:,i]) == 0:
                    img = scan_t1[:,:,i]
                else:
                    if conversion_no_bits == 8:
                        #img = gibbs_artifact_removal(img)
                        #img = scan_t1[:,:,i]
                        ##print('kakakaka')
                        ##print(np.min(img), np.max(img))
                        ##print(np.min(255*scan_t1[:,:,i]), np.max(255*scan_t1[:,:,i]))
                        img = np.uint8(255*(scan_t1[:,:,i]/np.max(scan_t1[:,:,i])))
                        #img=img
                        ##print(np.min(img), np.max(img))
                        #img = np.uint8(255*scan_t1[:,:,i]/np.max(scan_t1[:,:,i]))
                    elif conversion_no_bits == 16:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint16((2**16-1) * scan_t1[:,:,i]/np.max(scan_t1[:,:,i]))
                        img = img.astype(np.uint16)
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')
                
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1.png')

        elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
            scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t2 = scan_t2_loadfull.get_fdata()
            scan_t2_path = FILE_PATH

            # Biass Field removal:
            scan_t2 = gibbs_artifact_removal(scan_t2)

            # Save as PNG
            for i in range(scan_t2.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t2[:,:,i]) == 0:
                    img = scan_t2[:,:,i]
                else:
                    if conversion_no_bits == 8:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint8(255 * (scan_t2[:,:,i] / np.max(scan_t2[:,:,i])))
                    elif conversion_no_bits == 16:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint16((2**16-1) * scan_t2[:,:,i] / np.max(scan_t2[:,:,i]))
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't2.png')
        elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
            scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1ce = scan_t1ce_loadfull.get_fdata()
            scan_t1ce_path = FILE_PATH

            # Biass Field removal:
            scan_t1ce = gibbs_artifact_removal(scan_t1ce)

            # Save as PNG
            for i in range(scan_t1ce.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t1ce[:,:,i]) == 0:
                    img = scan_t1ce[:,:,i]
                else:
                    if conversion_no_bits == 8:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint8(255 * (scan_t1ce[:,:,i] / np.max(scan_t1ce[:,:,i])))
                    elif conversion_no_bits == 16:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint16((2**16-1) * scan_t1ce[:,:,i] / np.max(scan_t1ce[:,:,i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1ce.png')
        elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair': # in cazul in care este fisier flair
            scan_flair_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_flair = scan_flair_loadfull.get_fdata()
            scan_flair_path = FILE_PATH

            # Biass Field removal:
            scan_flair = gibbs_artifact_removal(scan_flair)

            # Save as PNG
            for i in range(scan_flair.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_flair[:,:,i]) == 0:
                    img = scan_flair[:,:,i]
                else:
                    if conversion_no_bits == 8:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint8(255 * (scan_flair[:,:,i] / np.max(scan_flair[:,:,i])))
                    elif conversion_no_bits == 16:
                        #img = gibbs_artifact_removal(img)
                        img = np.uint16((2**16-1) * scan_flair[:,:,i] / np.max(scan_flair[:,:,i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 'flair.png')
        elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg' and FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] != 'boolseg': # in cazul in care este fisier seg
            scan_seg_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_seg = scan_seg_loadfull.get_fdata()
            scan_seg_path = FILE_PATH
            # Save as PNG
            for i in range(scan_seg.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_seg[:, :, i]) == 0:
                    img = scan_seg[:, :, i]
                else:
                    img = scan_seg[:, :, i]
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * img / 4.0)
                        #img = np.uint8(255 * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 'seg.png')
        elif FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] == 'boolseg': # in cazul in care este fisier boolseg
            scan_boolseg_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_boolseg = scan_boolseg_loadfull.get_fdata()
            scan_boolseg_path = FILE_PATH
            # Save as PNG
            for i in range(scan_boolseg.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_boolseg[:, :, i]) == 0:
                    img = scan_boolseg[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_boolseg[:, :, i] / np.max(scan_boolseg[:, :, i]))
                    elif conversion_no_bits ==16:
                        img = np.uint16((2**16-1) * scan_boolseg[:, :, i] / np.max(scan_boolseg[:, :, i]))
                    else:
                        pass

                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 'boolseg.png')
