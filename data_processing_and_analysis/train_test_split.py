
'''
The script enters all all the files and sends 70% to the training folder, 20% to the val folser and 10% to the test folder
'''

import os
import gzip
import shutil

INPUT_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Training_Dataset\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
TRAINING_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Training_Dataset'
VAL_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Val_Datatset'
TEST_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\Test_Dataset'

i = 0
for dir_name_2 in os.listdir(INPUT_DIR_PATH):
    i += 1
    if i == 11:
        i = 1
    OLD_DIR_PATH = INPUT_DIR_PATH + '\\' + dir_name_2

    # calculare path noua locatie
    if i <= 7:
        NEW_DIR_PATH = TRAINING_DIR_PATH + '\\' + dir_name_2
        shutil.copytree(OLD_DIR_PATH, NEW_DIR_PATH)
    elif i<=9:
        NEW_DIR_PATH = VAL_DIR_PATH + '\\' + dir_name_2
        shutil.copytree(OLD_DIR_PATH, NEW_DIR_PATH)
    elif i==10:
        NEW_DIR_PATH = TEST_DIR_PATH + '\\' + dir_name_2
        shutil.copytree(OLD_DIR_PATH, NEW_DIR_PATH)
    else:
        #print("WARNING: unaccepted index.")

#print('End of process. Data has been split successfully')








