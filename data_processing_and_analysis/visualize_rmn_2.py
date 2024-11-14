'''

Script-ul va face un view cu T1 T2 T1CE FLAIR si segm pt un slice si va afisa si hitogrmalee

2 moduri:
0: fara percentile
1: cu percentile

'''

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from skimage import exposure
import cv2
import time

DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000'

for file_name in os.listdir(DIR_PATH):
    FILE_PATH = DIR_PATH + '\\' + file_name
    # #print(FILE_PATH)
    # se incarca fiecare fisier in obiectul corespunzator

    if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
        scan_t1 = nib.load(FILE_PATH).get_fdata()
        scan_t1_path = FILE_PATH
        #print(scan_t1.shape)
    elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2':  # in cazul in care este fisier t2
        scan_t2 = nib.load(FILE_PATH).get_fdata()
        scan_t2_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce':  # in cazul in care este fisier t1ce
        scan_t1ce = nib.load(FILE_PATH).get_fdata()
        scan_t1ce_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair':  # in cazul in care este fisier flair
        scan_flair = nib.load(FILE_PATH).get_fdata()
        scan_flair_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg':  # in cazul in care este fisier seg
        scan_seg = nib.load(FILE_PATH).get_fdata()
        scan_seg_path = FILE_PATH

# saving the visualised slices
VISUALIZED_SLICE = 25
#VISUALIZED_SLICE = 70
scan_t1_visualized_slice = scan_t1[:, :, VISUALIZED_SLICE]
scan_t2_visualized_slice = scan_t2[:, :, VISUALIZED_SLICE]
scan_t1ce_visualized_slice = scan_t1ce[:, :, VISUALIZED_SLICE]
scan_flair_visualized_slice = scan_flair[:, :, VISUALIZED_SLICE]
scan_seg_visualized_slice = scan_seg[:, :, VISUALIZED_SLICE]

# compute max and mins
scan_t1_min, scan_t1_max = np.min(scan_t1_visualized_slice), np.max(scan_t1_visualized_slice)
scan_t2_min, scan_t2_max = np.min(scan_t2_visualized_slice), np.max(scan_t2_visualized_slice)
scan_t1CE_min, scan_t1CE_max = np.min(scan_t1ce_visualized_slice), np.max(scan_t1ce_visualized_slice)
scan_FLAIR_min, scan_FLAIR_max = np.min(scan_flair_visualized_slice), np.max(scan_flair_visualized_slice)
scan_seg_min, scan_seg_max = np.min(scan_seg_visualized_slice), np.max(scan_seg_visualized_slice)

treshold = 50
# computing a histogram for the specific slice:
hist_scan_t1, hist_centers_scan_t1 = exposure.histogram(scan_t1_visualized_slice, nbins = 256)
hist_scan_t2, hist_centers_scan_t2 = exposure.histogram(scan_t2_visualized_slice, nbins = 256)
hist_scan_t1ce, hist_centers_scan_t1ce = exposure.histogram(scan_t1ce_visualized_slice, nbins = 256)
hist_scan_flair, hist_centers_scan_flair = exposure.histogram(scan_flair_visualized_slice, nbins = 256)
hist_scan_seg, hist_centers_scan_seg = exposure.histogram(scan_seg_visualized_slice, nbins = 256)

T1 = [scan_t1_visualized_slice, hist_centers_scan_t1, hist_scan_t1]
T2 = [scan_t2_visualized_slice, hist_centers_scan_t2, hist_scan_t2]
T1CE = [scan_t1ce_visualized_slice, hist_centers_scan_t1ce, hist_scan_t1ce]
FLAIR = [scan_flair_visualized_slice, hist_centers_scan_flair, hist_scan_flair]
SEG = [scan_seg_visualized_slice, hist_centers_scan_seg, hist_scan_seg]

par = {'axes.titlesize':9}
plt.rcParams.update(par)

fig, axes = plt.subplots(2, 5, figsize=(10, 10))
plt.subplots_adjust(wspace=0, hspace=0)

# plotare poze
ax1, ax2, ax3, ax4, ax5 = axes[0]
ax1.imshow(T1[0], cmap = 'gray')
ax1.axis('off') # for removing axis
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax1.set_title('T1')
ax2.imshow(T2[0], cmap = 'gray')
ax2.set_title('T2')
ax3.imshow(T1CE[0], cmap = 'gray')
ax3.set_title('T1CE')
ax4.imshow(FLAIR[0], cmap = 'gray')
ax4.set_title('FLAIR')
ax5.imshow(SEG[0], cmap = 'gray')
ax5.set_title('SEG:')

# plotare histograme
ax1, ax2, ax3, ax4, ax5 = axes[1]
ax1.plot(T1[1][1:], T1[2][1:])
ax1.set_title('T1: [' + "{:.2f}".format(scan_t1_min) + ", " + "{:.2f}".format(scan_t1_max) + ']')
ax2.plot(T2[1][1:], T2[2][1:])
ax2.set_title('T2: [' + "{:.2f}".format(scan_t2_min) + ", " + "{:.2f}".format(scan_t2_max) + ']')
ax3.plot(T1CE[1][1:], T1CE[2][1:])
ax3.set_title('T1CE: [' + "{:.2f}".format(scan_t1CE_min) + ", " + "{:.2f}".format(scan_t1CE_max) + ']')
ax4.plot(FLAIR[1][1:], FLAIR[2][1:])
ax4.set_title('FLAIR: [' + "{:.2f}".format(scan_FLAIR_min) + ", " + "{:.2f}".format(scan_FLAIR_max) + ']')
ax5.plot(SEG[1][1:], SEG[2][1:])
ax5.set_title('SEG: [' + "{:.2f}".format(scan_seg_min) + ", " + "{:.2f}".format(scan_seg_max) + ']')

plt.show()





'''
# VARIANATA ASTA MERGE, DAR CU PNG-uri IN LOC DE PLOT-uri
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from skimage import exposure
import cv2
import time

#print('Please insert mod: 0 = full hist; 1 = percentile')
mod = int(input())
#print(mod)
treshold = 90

DIR_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000'

for file_name in os.listdir(DIR_PATH):
    FILE_PATH = DIR_PATH + '\\' + file_name
    # #print(FILE_PATH)
    # se incarca fiecare fisier in obiectul corespunzator
    if FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't1':  # in cazul in care este fisier t1
        scan_t1 = nib.load(FILE_PATH).get_fdata()
        scan_t1_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-2:len(FILE_PATH.split('.')[0])] == 't2':  # in cazul in care este fisier t2
        scan_t2 = nib.load(FILE_PATH).get_fdata()
        scan_t2_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-4:len(FILE_PATH.split('.')[0])] == 't1ce':  # in cazul in care este fisier t1ce
        scan_t1ce = nib.load(FILE_PATH).get_fdata()
        scan_t1ce_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-5:len(FILE_PATH.split('.')[0])] == 'flair':  # in cazul in care este fisier flair
        scan_flair = nib.load(FILE_PATH).get_fdata()
        scan_flair_path = FILE_PATH
    elif FILE_PATH.split('.')[0][-3:len(FILE_PATH.split('.')[0])] == 'seg':  # in cazul in care este fisier seg
        scan_seg = nib.load(FILE_PATH).get_fdata()
        scan_seg_path = FILE_PATH

# saving the visualised slices
VISUALIZED_SLICE = 70
scan_t1_visualized_slice = scan_t1[:, :, VISUALIZED_SLICE]
scan_t2_visualized_slice = scan_t2[:, :, VISUALIZED_SLICE]
scan_t1ce_visualized_slice = scan_t1ce[:, :, VISUALIZED_SLICE]
scan_flair_visualized_slice = scan_flair[:, :, VISUALIZED_SLICE]
scan_seg_visualized_slice = scan_seg[:, :, VISUALIZED_SLICE]

# compute max and mins
scan_t1_min, scan_t1_max = np.min(scan_t1_visualized_slice), np.max(scan_t1_visualized_slice)
scan_t2_min, scan_t2_max = np.min(scan_t2_visualized_slice), np.max(scan_t2_visualized_slice)
scan_t1CE_min, scan_t1CE_max = np.min(scan_t1ce_visualized_slice), np.max(scan_t1ce_visualized_slice)
scan_FLAIR_min, scan_FLAIR_max = np.min(scan_flair_visualized_slice), np.max(scan_flair_visualized_slice)
scan_seg_min, scan_seg_max = np.min(scan_seg_visualized_slice), np.max(scan_seg_visualized_slice)

treshold = 50
# computing a histogram for the specific slice:
hist_scan_t1, hist_centers_scan_t1 = exposure.histogram(scan_t1_visualized_slice, nbins = 256)
if mod == 1:
    percentile = np.percentile(np.array(hist_scan_t1), treshold)
    hist_scan_t1 = np.where(hist_scan_t1 > percentile)[0]
    #print(hist_scan_t1)
    #print(len(hist_scan_t1))
    hist_centers_scan_t1 = np.array(range(len(hist_scan_t1)))
    #print(len(hist_centers_scan_t1))
hist_scan_t2, hist_centers_scan_t2 = exposure.histogram(scan_t2_visualized_slice, nbins = 256)
hist_scan_t1ce, hist_centers_scan_t1ce = exposure.histogram(scan_t1ce_visualized_slice, nbins = 256)
hist_scan_flair, hist_centers_scan_flair = exposure.histogram(scan_flair_visualized_slice, nbins = 256)
hist_scan_seg, hist_centers_scan_seg = exposure.histogram(scan_seg_visualized_slice, nbins = 256)

# save histogram plots
plt.plot(hist_centers_scan_t1, hist_scan_t1)
plt.savefig(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t1.png', transparent = True, dpi = 1000)
picturehist_scan_t1 = np.array(cv2.imread(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t1.png'))
plt.show()
time.sleep(20)
plt.clf()
plt.plot(hist_centers_scan_t2, hist_scan_t2)
plt.savefig(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t2.png', transparent = True, dpi = 1000)
picturehist_scan_t2 = np.array(cv2.imread(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t2.png'))
#plt.show()
#time.sleep(20)
plt.clf()
plt.plot(hist_centers_scan_t1ce, hist_scan_t1ce)
plt.savefig(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t1ce.png', transparent = True, dpi = 1000)
picturehist_scan_t1CE = np.array(cv2.imread(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_t1ce.png'))
#plt.show()
#time.sleep(20)
plt.clf()
plt.plot(hist_centers_scan_flair, hist_scan_flair)
plt.savefig(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_flair.png', transparent = True, dpi = 1000)
picturehist_scan_FLAIR = np.array(cv2.imread(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_flair.png'))
#plt.show()
#time.sleep(20)
plt.clf()
plt.plot(hist_centers_scan_seg, hist_scan_seg)
plt.savefig(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_seg.png', transparent = True)
picturehist_scan_seg = np.array(cv2.imread(r'E:\an_4_LICENTA\Workspace\junkdata\hist_centers_scan_seg.png'))
plt.clf()

T1 = [scan_t1_visualized_slice, picturehist_scan_t1]
T2 = [scan_t2_visualized_slice, picturehist_scan_t2]
T1CE = [scan_t1ce_visualized_slice, picturehist_scan_t1CE]
FLAIR = [scan_flair_visualized_slice, picturehist_scan_FLAIR]
SEG = [scan_seg_visualized_slice, picturehist_scan_seg]

fig, axes = plt.subplots(2, 5, figsize=(10,10))
plt.subplots_adjust(wspace=0, hspace=0)

par = {'axes.titlesize':10}
plt.rcParams.update(par)

fig, axes = plt.subplots(2, 5, figsize=(10,10))
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(2):
    ax1, ax2, ax3, ax4, ax5 = axes[i]
    ax1.imshow(T1[i], cmap = 'gray')
    ax1.axis('off') # for removing axis
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax1.set_title('T1: [' + str(scan_t1_min) + ", " + str(scan_t1_max) + ']')
    ax2.imshow(T2[i], cmap = 'gray')
    ax2.set_title('T2: [' + str(scan_t2_min) + ", " + str(scan_t2_max) + ']')
    ax3.imshow(T1CE[i], cmap = 'gray')
    ax3.set_title('T1CE: [' + str(scan_t1CE_min) + ", " + str(scan_t1CE_max) + ']')
    ax4.imshow(FLAIR[i], cmap = 'gray')
    ax4.set_title('FLAIR: [' + str(scan_FLAIR_min) + ", " + str(scan_FLAIR_max) + ']')
    ax5.imshow(SEG[i], cmap = 'gray')
    ax5.set_title('SEG: [' + str(scan_seg_min) + ", " + str(scan_seg_max) + ']')
plt.show()
'''
'''
fig, axes = plt.subplots(2, 5, figsize=(10,10))
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(2):
    ax1, ax2, ax3, ax4, ax5 = axes[i]
    #ax1.imshow(T1[i], cmap = 'gray')
    ax1.imshow(T1[0], cmap='gray')
    ax1.plot(T1[2], T1[1], 'b')
    ax1.axis('off') # for removing axis
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax1.set_title('T1')
    ax2.imshow(T2[i], cmap = 'gray')
    ax2.set_title('T2')
    ax3.imshow(T1CE[i], cmap = 'gray')
    ax3.set_title('T1CE')
    ax4.imshow(FLAIR[i], cmap = 'gray')
    ax4.set_title('FLAIR')
    ax5.imshow(SEG[i], cmap = 'gray')
    ax5.set_title('SEG')
plt.show()
'''


