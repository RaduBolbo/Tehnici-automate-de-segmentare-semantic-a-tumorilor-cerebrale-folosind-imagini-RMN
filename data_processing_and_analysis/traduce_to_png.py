'''
The script will traduce all .nii files to a number of pngs equal to the number of slices*5

Usage:
INPUT_PATH = the path to a dir containing all the scans
OUTPUT_PATH = the path where the .png files will be saved, each in a new dir named as in INPUT_PATH

'''



import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
import cv2 as cv

# pentru setul de date de testare
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Test_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Test_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\16b_PNG_Test_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Test_Dataset'

# Pentru setul de date de tarin
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Reduced_Dataset\PNG_Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\16b_PNG_Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Train_Dataset'

# Pentru setul de date de val
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Val_Dataset'

# pentru setul de OFICIAL DE VALIDARE OFICIAL ----------------------------
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\PNG_Val_Official_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\16b_PNG_Val_Dataset'
OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\PNG_Val_Official_Dataset_Normalizare_Globala'

# Ce se intampla grwit cu GT-ul?
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla'

#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_fara_gz'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_traduced_to_PNG_from_pas1'

#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\tst_in'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\tst_out' # normalizare doar pe slice-ul curent
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\tsts_out_normalizarepetotscanul' # normalizare pe tot scan-ul


conversion_no_bits = 16

for dir_name_2 in os.listdir(INPUT_PATH):
    INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
    OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2
    '''
    if os.path.isdir(OUTPUT_PATH_2) == False:
        os.mkdir(OUTPUT_PATH_2)
    '''
    print('dir: ' + dir_name_2)

    for file_name in os.listdir(INPUT_PATH_2):
        FILE_PATH = INPUT_PATH_2 + '\\' + file_name
        OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '_slice_'
        #print('file: ' + file_name)

        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
            #print('este t1')
            
            scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1 = scan_t1_loadfull.get_fdata()
            scan_t1_path = FILE_PATH
            

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
                        img = np.uint8(255*scan_t1[:,:,i]/np.max(scan_t1[:,:,i])) # normalizare doar pe slice-ul curent
                        #img = np.uint8(255*scan_t1[:,:,i]/np.max(scan_t1[:,:,:])) # clacul global, ep tot scan-ul
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t1[:,:,i]/np.max(scan_t1[:,:,:])) # clacul global, ep tot scan-ul
                        img = img.astype(np.uint16)
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')

                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1.png')

        elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
            #print('este t2')

            scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t2 = scan_t2_loadfull.get_fdata()
            scan_t2_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t2.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t2[:, :, i]) == 0:
                    img = scan_t2[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t2[:, :, i] / np.max(scan_t2[:, :, i])) # normalizare doar pe slice-ul curent
                        #img = np.uint8(255 * scan_t2[:, :, i] / np.max(scan_t2[:, :, :])) # clacul global, ep tot scan-ul
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t2[:, :, i] / np.max(scan_t2[:, :, :])) # clacul global, ep tot scan-ul
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't2.png')
        elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
            #print('este t1ce')

            scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1ce = scan_t1ce_loadfull.get_fdata()
            scan_t1ce_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t1ce.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t1ce[:, :, i]) == 0:
                    img = scan_t1ce[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i])) # normalizare doar pe slice-ul curent
                        #img = np.uint8(255 * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, :])) # clacul global, ep tot scan-ul
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, :])) # clacul global, ep tot scan-ul
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1ce.png')
        elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair': # in cazul in care este fisier flair
            #print('este flair')

            scan_flair_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_flair = scan_flair_loadfull.get_fdata()
            scan_flair_path = FILE_PATH
            # Save as PNG
            for i in range(scan_flair.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_flair[:, :, i]) == 0:
                    img = scan_flair[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        
                        #print('before: ')
                        #u, c = np.unique(scan_flair[:, :, i], return_counts=True)
                        #print(dict(zip(u, c)))
                        img = np.uint8(255 * scan_flair[:, :, i] / np.max(scan_flair[:, :, i])) # normalizare doar pe slice-ul curent
                        #img = np.uint8(255 * scan_flair[:, :, i] / np.max(scan_flair[:, :, :])) # clacul global, ep tot scan-ul
                        #print('after: ')
                        #u, c = np.unique(img, return_counts=True)
                        #rint(dict(zip(u, c)))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_flair[:, :, i] / np.max(scan_flair[:, :, :])) # clacul global, ep tot scan-ul
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 'flair.png')
        elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg' and FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] != 'boolseg': # in cazul in care este fisier seg
            #print('este seg')
            
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
                    #print('intru in if: ' + str(i))

                    img = scan_seg[:, :, i]
                else:
                    #print('intru in else: ' + str(i))
                    img = scan_seg[:, :, i]

                    if conversion_no_bits == 8:
                        #print('before: ')
                        #u, c = np.unique(img, return_counts=True)
                        #print(dict(zip(u, c)))
                        #img = np.uint8(255 * img / np.max(scan_seg[:, :, i]))
                        ################################################### EXACT AICI SE PETRECEA EROAREA ###########################################################################
                        img = np.uint8(255 * img / 4.0) # 4.0, deoarece 4.0 este maximul posibil
                        #print('after: ')
                        #u, c = np.unique(img, return_counts=True)
                        #print(dict(zip(u, c)))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    #print(type(img))
                    #print(img)
                    img = img.convert('L') # cred ca atunci cand e o singura clasa, o valoare oricat de mica este convertita la alb
                img.save(SLICE_DIR + '\\' + 'seg.png')
        elif FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] == 'boolseg': # in cazul in care este fisier boolseg
            #print('este boolseg')

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

"""
###########################################################################################
################### ASTA ARE CORECTATA PROBLEMA CU SALTUL DE CLASE ########################
###########################################################################################
for dir_name_2 in os.listdir(INPUT_PATH):
    INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
    OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2
    '''
    if os.path.isdir(OUTPUT_PATH_2) == False:
        os.mkdir(OUTPUT_PATH_2)
    '''
    print('dir: ' + dir_name_2)

    for file_name in os.listdir(INPUT_PATH_2):
        FILE_PATH = INPUT_PATH_2 + '\\' + file_name
        OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '_slice_'
        print('file: ' + file_name)

        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
            print('este t1')
            
            scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1 = scan_t1_loadfull.get_fdata()
            scan_t1_path = FILE_PATH
            

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
                        img = np.uint8(255*scan_t1[:,:,i]/np.max(scan_t1[:,:,i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t1[:,:,i]/np.max(scan_t1[:,:,i]))
                        img = img.astype(np.uint16)
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')

                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1.png')

        elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
            print('este t2')

            scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t2 = scan_t2_loadfull.get_fdata()
            scan_t2_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t2.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t2[:, :, i]) == 0:
                    img = scan_t2[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t2[:, :, i] / np.max(scan_t2[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t2[:, :, i] / np.max(scan_t2[:, :, i]))
                    else:
                        print('You must enter a conversion_no_bits of 8 or 16')
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't2.png')
        elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
            print('este t1ce')

            scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1ce = scan_t1ce_loadfull.get_fdata()
            scan_t1ce_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t1ce.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t1ce[:, :, i]) == 0:
                    img = scan_t1ce[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 't1ce.png')
        elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair': # in cazul in care este fisier flair
            print('este flair')

            scan_flair_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_flair = scan_flair_loadfull.get_fdata()
            scan_flair_path = FILE_PATH
            # Save as PNG
            for i in range(scan_flair.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_flair[:, :, i]) == 0:
                    img = scan_flair[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_flair[:, :, i] / np.max(scan_flair[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_flair[:, :, i] / np.max(scan_flair[:, :, i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    img = img.convert('L')
                img.save(SLICE_DIR + '\\' + 'flair.png')
        elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg' and FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] != 'boolseg': # in cazul in care este fisier seg
            print('este seg')
            
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
                    print('intru in if: ' + str(i))

                    img = scan_seg[:, :, i]
                else:
                    print('intru in else: ' + str(i))
                    img = scan_seg[:, :, i]

                    if conversion_no_bits == 8:
                        print('before: ')
                        #print(np.unique(img, return_counts=True))
                        u, c = np.unique(img, return_counts=True)
                        print(dict(zip(u, c)))
                        #img = np.uint8(255 * img / np.max(scan_seg[:, :, i]))
                        ################################################### EXACT AICI SE PETRECEA EROAREA ###########################################################################
                        img = np.uint8(255 * img / 4.0) # 4.0, deoarece 4.0 este maximul posibil
                        print('after: ')
                        u, c = np.unique(img, return_counts=True)
                        print(dict(zip(u, c)))
                        #img = np.uint8(255 * img)
                    elif conversion_no_bits == 16:
                        #print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
                        img = np.uint16((2**16-1) * scan_seg[:, :, i] / np.max(scan_seg[:, :, i]))
                    else:
                        pass
                img = Image.fromarray(img)
                if conversion_no_bits == 8:
                    #print(type(img))
                    #print(img)
                    img = img.convert('L') # cred ca atunci cand e o singura clasa, o valoare oricat de mica este convertita la alb
                img.save(SLICE_DIR + '\\' + 'seg.png')
        elif FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] == 'boolseg': # in cazul in care este fisier boolseg
            print('este boolseg')

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
"""

"""
##############################################################################################
############################ ASTA E VARINAT FOLOSITA IN GENERAL ##############################
##############################################################################################
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
                        img = np.uint8(255*scan_t1[:,:,i]/np.max(scan_t1[:,:,i]))
                    elif conversion_no_bits == 16:
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
            # Save as PNG
            for i in range(scan_t2.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t2[:, :, i]) == 0:
                    img = scan_t2[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t2[:, :, i] / np.max(scan_t2[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t2[:, :, i] / np.max(scan_t2[:, :, i]))
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
            # Save as PNG
            for i in range(scan_t1ce.shape[2]):
                # creadre directoare slice x, daca nu exista deja
                SLICE_DIR = OUTPUT_FILE_PATH + str(i)
                if os.path.isdir(SLICE_DIR) == False:
                    os.mkdir(SLICE_DIR)

                if np.max(scan_t1ce[:, :, i]) == 0:
                    img = scan_t1ce[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i]))
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

                if np.max(scan_flair[:, :, i]) == 0:
                    img = scan_flair[:, :, i]
                else:
                    if conversion_no_bits == 8:
                        img = np.uint8(255 * scan_flair[:, :, i] / np.max(scan_flair[:, :, i]))
                    elif conversion_no_bits == 16:
                        img = np.uint16((2**16-1) * scan_flair[:, :, i] / np.max(scan_flair[:, :, i]))
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

"""

'''

for dir_name_2 in os.listdir(INPUT_PATH):
    INPUT_PATH_2 = INPUT_PATH + '\\' + dir_name_2
    OUTPUT_PATH_2 = OUTPUT_PATH + '\\' + dir_name_2
    if os.path.isdir(OUTPUT_PATH_2) == False:
        os.mkdir(OUTPUT_PATH_2)

    for file_name in os.listdir(INPUT_PATH_2):
        FILE_PATH = INPUT_PATH_2 + '\\' + file_name
        OUTPUT_FILE_PATH = OUTPUT_PATH_2 + '\\' + file_name
        ##print(FILE_PATH)
        # se incarca fiecare fisier in obiectul corespunzator
        if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
            scan_t1_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1 = scan_t1_loadfull.get_fdata()
            scan_t1_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t1.shape[2]):
                if np.max(scan_t1[:,:,i]) == 0:
                    img = scan_t1[:,:,i]
                else:
                    img = 255*scan_t1[:,:,i]/np.max(scan_t1[:,:,i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')
        elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2': # in cazul in care este fisier t2
            scan_t2_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t2 = scan_t2_loadfull.get_fdata()
            scan_t2_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t2.shape[2]):
                if np.max(scan_t2[:, :, i]) == 0:
                    img = scan_t2[:, :, i]
                else:
                    img = 255 * scan_t2[:, :, i] / np.max(scan_t2[:, :, i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')
        elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce': # in cazul in care este fisier t1ce
            scan_t1ce_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_t1ce = scan_t1ce_loadfull.get_fdata()
            scan_t1ce_path = FILE_PATH
            # Save as PNG
            for i in range(scan_t1ce.shape[2]):
                if np.max(scan_t1ce[:, :, i]) == 0:
                    img = scan_t1ce[:, :, i]
                else:
                    img = 255 * scan_t1ce[:, :, i] / np.max(scan_t1ce[:, :, i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')
        elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair': # in cazul in care este fisier flair
            scan_flair_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_flair = scan_flair_loadfull.get_fdata()
            scan_flair_path = FILE_PATH
            # Save as PNG
            for i in range(scan_flair.shape[2]):
                if np.max(scan_flair[:, :, i]) == 0:
                    img = scan_flair[:, :, i]
                else:
                    img = 255 * scan_flair[:, :, i] / np.max(scan_flair[:, :, i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')
        elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg': # in cazul in care este fisier seg
            scan_seg_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_seg = scan_seg_loadfull.get_fdata()
            scan_seg_path = FILE_PATH
            # Save as PNG
            for i in range(scan_seg.shape[2]):
                if np.max(scan_seg[:, :, i]) == 0:
                    img = scan_seg[:, :, i]
                else:
                    img = 255 * scan_seg[:, :, i] / np.max(scan_seg[:, :, i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')
        elif FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] == 'boolseg': # in cazul in care este fisier boolseg
            scan_boolseg_loadfull = nib.load(FILE_PATH, mmap=False)
            scan_boolseg = scan_boolseg_loadfull.get_fdata()
            scan_boolseg_path = FILE_PATH
            # Save as PNG
            for i in range(scan_boolseg.shape[2]):
                if np.max(scan_boolseg[:, :, i]) == 0:
                    img = scan_boolseg[:, :, i]
                else:
                    img = 255 * scan_boolseg[:, :, i] / np.max(scan_boolseg[:, :, i])
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(OUTPUT_FILE_PATH + '_slice_' + str(i) + '.png')

'''