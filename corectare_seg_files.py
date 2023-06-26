'''
Se aplica pe acele directoare un de am gresit fisierele de segmentare
'''

import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
import cv2 as cv

########### CORECTAT

#Train
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Train_Dataset'

#Val
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Val_Dataset'

# Test
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Test_Dataset'
#OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Normalizare_globala\16b_PNG_Test_Dataset'




conversion_no_bits = 8

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

        
        if FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg' and FILE_PATH.split('.')[0][-7:len(FILE_PATH.split('.')[0])] != 'boolseg': # in cazul in care este fisier seg
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
        
















