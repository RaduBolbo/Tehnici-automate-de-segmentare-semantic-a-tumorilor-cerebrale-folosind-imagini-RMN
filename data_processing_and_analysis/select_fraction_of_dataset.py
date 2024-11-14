'''
The script enters all all the files and sends 70% to the training folder, 20% to the val folser and 10% to the test folder
'''

import os
import gzip
import shutil

#INPUT_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Training_Dataset'
#REDUCED_TRAINING_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'


# Train
INPUT_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'
REDUCED_TRAINING_DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Train_Dataset'


i = 0
for dir_name_2 in os.listdir(INPUT_DIR_PATH):
    i += 1
    if i == 4:
        i = 1
    OLD_DIR_PATH = INPUT_DIR_PATH + '\\' + dir_name_2

    # calculare path noua locatie
    if i == 2:
        NEW_DIR_PATH = REDUCED_TRAINING_DIR_PATH + '\\' + dir_name_2
        shutil.copytree(OLD_DIR_PATH, NEW_DIR_PATH)

#print('End of process. Data has been split successfully')
