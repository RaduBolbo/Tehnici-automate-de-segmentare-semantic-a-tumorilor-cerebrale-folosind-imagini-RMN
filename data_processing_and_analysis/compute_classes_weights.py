




'''
The script enters all te examples in the given folder and counts
tyhe number of voxels in each class in the "seg" files
'''
import os
import numpy as np
import cv2
import scipy

DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Training_Dataset'

# creez 3 liste pt ED, NCR, ET: fiecare lista contine, ca elemente, nuamrul de voxeli atributiti in acea clasa pentru un anumit exemplu
list_ED = []
list_ET = []
list_NCR = []

# iterez prin director
for dir_name in os.listdir(DIR_PATH):
    # incarc imaginea cui segmenatrea
    labels = cv2.imread(DIR_PATH + '\\' + dir_name + '\\' + 'seg.png')
    
    # Extrag GT-urile celor 3 canale:
    target_ET = np.where(labels[:, :, 0] == 255, 1, 0) # 4 este label-ul ET
    target_NCR = np.where(labels[:, :, 0] == 63, 1, 0)
    target_ED = np.where(labels[:, :, 0] == 127, 1, 0) # toate label-urile nenume reprezinta WT

    list_ED.append(np.count_nonzero(target_ED))
    list_ET.append(np.count_nonzero(target_ET))
    list_NCR.append(np.count_nonzero(target_NCR))


nr_ED_volxels = np.sum(list_ED)
nr_ET_volxels = np.sum(list_ET)
nr_NCR_volxels = np.sum(list_NCR)

W_ED = (nr_ED_volxels + nr_ET_volxels + nr_NCR_volxels)/nr_ED_volxels
W_ET = (nr_ED_volxels + nr_ET_volxels + nr_NCR_volxels)/nr_ET_volxels
W_NCR = (nr_ED_volxels + nr_ET_volxels + nr_NCR_volxels)/nr_NCR_volxels

#print(W_ED)
#print(W_ET)
#print(W_NCR)

W_ED, W_ET, W_NCR = scipy.special.softmax([W_ED, W_ET, W_NCR])
'''
#print(nr_ED_volxels)
#print(nr_ET_volxels)
#print(nr_NCR_volxels)
'''
#print(W_ED)
#print(W_ET)
#print(W_NCR)
















