
'''
The script enters all te examples in the given folder and decompresses each file
'''

import os
import gzip
import shutil

#DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\Train_fara_gz'
DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Z_ceseintampla\tst_in'

######################
# 1) se dezarhiveaza .gz-urile
######################

for dir_name_2 in os.listdir(DIR_PATH):
    DIR_PATH_2 = DIR_PATH + '\\' + dir_name_2
    for file_name in os.listdir(DIR_PATH_2):
        FILE_PATH = DIR_PATH_2 + '\\' + file_name
        NEW_FILE_PATH = FILE_PATH[:-3]
        if FILE_PATH.split('.')[-1] == 'gz':
            with gzip.open(FILE_PATH, 'rb') as f_in:
                with open(NEW_FILE_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

######################
# 2) se sterg .gz-urile
######################

for dir_name_2 in os.listdir(DIR_PATH):
    DIR_PATH_2 = DIR_PATH + '\\' + dir_name_2
    for file_name in os.listdir(DIR_PATH_2):
        FILE_PATH = DIR_PATH_2 + '\\' + file_name
        if FILE_PATH.split('.')[-1] == 'gz':
            os.remove(FILE_PATH)


