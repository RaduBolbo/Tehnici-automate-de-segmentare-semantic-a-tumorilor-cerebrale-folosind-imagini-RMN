
'''

Acest script va folosi implementarea lui Nyul

'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import SimpleITK as sitk
from nipype.interfaces.ants import N4BiasFieldCorrection
from skimage import exposure
import skimage
import os
from PIL import Image

# pentru setul de date de validare
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Test_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\biassfieldremoved_PNG_Test_Dataset'

# Pentru setul de date de tarin
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\biassfieldremoved_PNG_Train_Dataset'

# pentru setul de date de validare
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Val_Dataset'
fraction = False

#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Train_Dataset'
#fraction = True

conversion_no_bits = 8

############
# Definire functii specifice bias field removal
############

def min_max_normalization(img):
    min = np.min(img)
    max = np.max(img)
    normalized_image = (img - min) / (max - min)
    final_image = sitk.GetImageFromArray(normalized_image)
    return final_image

def extend_to_0_255(img):
    img = sitk.GetArrayFromImage(img)
    min = np.min(img)
    max = np.max(img)
    rescaled_image = (img - min)*256 / (max - min)
    return np.uint8(rescaled_image)

def n4_bias_correction(image):
    '''
        PARTAMETRII MENIONATI IN APPERUL:
        https://bmjopen.bmj.com/content/bmjopen/12/7/e059000.full.pdf
    '''

    ##print(type(image))
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = 'structural.nii'
    n4.inputs.bspline_fitting_distance = 100
    n4.inputs.shrink_factor = 2
    n4.inputs.inputs.convergence_threshold = 0
    n4.inputs.n_iterations = [20,20,20,10]
    n4.cmdline

    return output_corrected

def corectie_bias_slice(img):
    normalized_image = min_max_normalization(img)
    bias_corrected = n4_bias_correction(normalized_image)
    bias_corrected_restored = extend_to_0_255(bias_corrected)
    return bias_corrected_restored

if fraction == True:

    for dir_name_2 in os.listdir(INPUT_PATH):
        INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
        OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2

        #print(dir_name_2)
        for file_name in os.listdir(INPUT_PATH_2):
            FILE_PATH = INPUT_PATH_2 + '\\' + file_name
            OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '_slice_'

            # se incarca fiecare fisier in obiectul corespunzator
            if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
                scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t1 = scan_t1_loadfull.get_fdata()
                scan_t1_path = FILE_PATH

                # Save as PNG
                j = 0
                for i in range(scan_t1.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
                        # creadre directoare slice x, daca nu exista deja
                        SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                        if os.path.isdir(SLICE_DIR) == False:
                            os.mkdir(SLICE_DIR)

                        # Biass Field removal:
                        img = corectie_bias_slice(scan_t1[:,:,i])

                        if np.max(img) == 0:
                            img = img
                        else:
                            if conversion_no_bits == 8:
                                img = img # V2
                                #img = np.uint8(255*(img/np.max(img))) # V1
                            elif conversion_no_bits == 16:
                                #img = corectie_bias_slice(img)
                                img = np.uint16((2**16-1) * img/np.max(img))
                                img = img.astype(np.uint16)
                            else:
                                #print('You must enter a conversion_no_bits of 8 or 16')
                    
                        img = Image.fromarray(img)
                        if conversion_no_bits == 8:
                            img = img.convert('L')
                        img.save(SLICE_DIR + '\\' + 't1.png')

            elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
                scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t2 = scan_t2_loadfull.get_fdata()
                scan_t2_path = FILE_PATH

                # Save as PNG
                j = 0
                for i in range(scan_t2.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
                        # creadre directoare slice x, daca nu exista deja
                        SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                        if os.path.isdir(SLICE_DIR) == False:
                            os.mkdir(SLICE_DIR)

                        # Biass Field removal:
                        img = corectie_bias_slice(scan_t2[:,:,i])

                        if np.max(img) == 0:
                            img = img
                        else:
                            if conversion_no_bits == 8:
                                #img = corectie_bias_slice(img)
                                img = img # V2
                                #img = np.uint8(255 * (img / np.max(img))) # V1
                            elif conversion_no_bits == 16:
                                #img = corectie_bias_slice(img)
                                img = np.uint16((2**16-1) * img / np.max(img))
                            else:
                                #print('You must enter a conversion_no_bits of 8 or 16')
                        img = Image.fromarray(img)
                        if conversion_no_bits == 8:
                            img = img.convert('L')
                        img.save(SLICE_DIR + '\\' + 't2.png')
            elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
                scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t1ce = scan_t1ce_loadfull.get_fdata()
                scan_t1ce_path = FILE_PATH

                # Save as PNG
                j = 0
                for i in range(scan_t1ce.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
                        # creadre directoare slice x, daca nu exista deja
                        SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                        if os.path.isdir(SLICE_DIR) == False:
                            os.mkdir(SLICE_DIR)

                        # Biass Field removal:
                        img = corectie_bias_slice(scan_t1ce[:,:,i])

                        if np.max(img) == 0:
                            img = img
                        else:
                            if conversion_no_bits == 8:
                                #img = corectie_bias_slice(img)
                                img = img # V2
                                #img = np.uint8(255 * (img / np.max(img))) # V1
                            elif conversion_no_bits == 16:
                                #img = corectie_bias_slice(img)
                                img = np.uint16((2**16-1) * img / np.max(img))
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

                # Save as PNG
                j = 0
                for i in range(scan_flair.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
                        # creadre directoare slice x, daca nu exista deja
                        SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                        if os.path.isdir(SLICE_DIR) == False:
                            os.mkdir(SLICE_DIR)
                        
                        # Biass Field removal:
                        img = corectie_bias_slice(scan_flair[:,:,i])
                        
                        if np.max(img) == 0:
                            img = img
                        else:
                            if conversion_no_bits == 8:
                                #img = corectie_bias_slice(img)
                                img = img # V2
                                #img = np.uint8(255 * (img / np.max(img))) # V1
                            elif conversion_no_bits == 16:
                                #img = corectie_bias_slice(img)
                                img = np.uint16((2**16-1) * img / np.max(img))
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
                j = 0
                for i in range(scan_seg.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
                        # creadre directoare slice x, daca nu exista deja
                        SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                        if os.path.isdir(SLICE_DIR) == False:
                            os.mkdir(SLICE_DIR)

                        if np.max(scan_seg[:, :, i]) == 0:
                            img = scan_seg[:, :, i]
                        else:
                            if conversion_no_bits == 8:
                                img = np.uint8(255 * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
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
                j = 0
                for i in range(scan_boolseg.shape[2]):
                    j += 1
                    if j == 4:
                        j = 1
                    if j == 2:
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
else:
    for dir_name_2 in os.listdir(INPUT_PATH):
        INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
        OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2

        #print(dir_name_2)
        for file_name in os.listdir(INPUT_PATH_2):
            FILE_PATH = INPUT_PATH_2 + '\\' + file_name
            OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '_slice_'

            # se incarca fiecare fisier in obiectul corespunzator
            if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
                scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t1 = scan_t1_loadfull.get_fdata()
                scan_t1_path = FILE_PATH

                # Save as PNG
                for i in range(scan_t1.shape[2]):
                    # creadre directoare slice x, daca nu exista deja
                    SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                    if os.path.isdir(SLICE_DIR) == False:
                        os.mkdir(SLICE_DIR)

                    # Biass Field removal:
                    img = corectie_bias_slice(scan_t1[:,:,i])

                    if np.max(img) == 0:
                        img = img
                    else:
                        if conversion_no_bits == 8:
                            img = img # V2
                            #img = np.uint8(255*(img/np.max(img))) # V1
                        elif conversion_no_bits == 16:
                            #img = corectie_bias_slice(img)
                            img = np.uint16((2**16-1) * img/np.max(img))
                            img = img.astype(np.uint16)
                        else:
                            #print('You must enter a conversion_no_bits of 8 or 16')
                    
                    img = Image.fromarray(img)
                    if conversion_no_bits == 8:
                        img = img.convert('L')
                    img.save(SLICE_DIR + '\\' + 't1.png')

            elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
                scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t2 = scan_t2_loadfull.get_fdata()
                scan_t2_path = FILE_PATH

                # Save as PNG
                for i in range(scan_t2.shape[2]):
                    # creadre directoare slice x, daca nu exista deja
                    SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                    if os.path.isdir(SLICE_DIR) == False:
                        os.mkdir(SLICE_DIR)

                    # Biass Field removal:
                    img = corectie_bias_slice(scan_t2[:,:,i])

                    if np.max(img) == 0:
                        img = img
                    else:
                        if conversion_no_bits == 8:
                            #img = corectie_bias_slice(img)
                            img = img # V2
                            #img = np.uint8(255 * (img / np.max(img))) # V1
                        elif conversion_no_bits == 16:
                            #img = corectie_bias_slice(img)
                            img = np.uint16((2**16-1) * img / np.max(img))
                        else:
                            #print('You must enter a conversion_no_bits of 8 or 16')
                    img = Image.fromarray(img)
                    if conversion_no_bits == 8:
                        img = img.convert('L')
                    img.save(SLICE_DIR + '\\' + 't2.png')
            elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
                scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
                scan_t1ce = scan_t1ce_loadfull.get_fdata()
                scan_t1ce_path = FILE_PATH

                # Save as PNG
                for i in range(scan_t1ce.shape[2]):
                    # creadre directoare slice x, daca nu exista deja
                    SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                    if os.path.isdir(SLICE_DIR) == False:
                        os.mkdir(SLICE_DIR)

                    # Biass Field removal:
                    img = corectie_bias_slice(scan_t1ce[:,:,i])

                    if np.max(img) == 0:
                        img = img
                    else:
                        if conversion_no_bits == 8:
                            #img = corectie_bias_slice(img)
                            img = img # V2
                            #img = np.uint8(255 * (img / np.max(img))) # V1
                        elif conversion_no_bits == 16:
                            #img = corectie_bias_slice(img)
                            img = np.uint16((2**16-1) * img / np.max(img))
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

                # Save as PNG
                for i in range(scan_flair.shape[2]):
                    # creadre directoare slice x, daca nu exista deja
                    SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                    if os.path.isdir(SLICE_DIR) == False:
                        os.mkdir(SLICE_DIR)
                    
                    # Biass Field removal:
                    img = corectie_bias_slice(scan_flair[:,:,i])
                    
                    if np.max(img) == 0:
                        img = img
                    else:
                        if conversion_no_bits == 8:
                            #img = corectie_bias_slice(img)
                            img = img # V2
                            #img = np.uint8(255 * (img / np.max(img))) # V1
                        elif conversion_no_bits == 16:
                            #img = corectie_bias_slice(img)
                            img = np.uint16((2**16-1) * img / np.max(img))
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
                        if conversion_no_bits == 8:
                            img = np.uint8(255 * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
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

