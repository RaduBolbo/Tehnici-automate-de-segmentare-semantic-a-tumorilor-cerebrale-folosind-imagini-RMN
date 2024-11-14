import numpy as np
import cv2 as cv
import nibabel as nib

# importr din algoritmul scris de RafaelNH:
from gibbs_artifact_removal_by_RafaelNH import _gibbs_removal_2d
from gibbs_artifact_removal_by_RafaelNH import gibbs_removal

####
# definire functie
####

def gibbs_removal_2D(img):

    corrected_img = _gibbs_removal_2d(img[:,:,0])

    return corrected_img


def gibbs_removal_3D(scan):

    #corrected_scan = gibbs_removal(scan, slice_axis=2, n_points=3)
    corrected_scan = gibbs_removal(scan, slice_axis=2, n_points=5)

    return corrected_scan

INPUT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Training_Dataset\BraTS2021_00002_slice_31\t1.png"
OUTPUT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Gibs_artifact_test"

################################## ETAPA 1. INCERCARE 2D ########################################

####
# Incaracre poze
####

img = cv.imread(INPUT_PATH)

####
# artefact removal
####

corrected_img = gibbs_removal_2D(img)

####
# Salvare poze
####

cv.imwrite(OUTPUT_PATH + "\\" + 'original_2D.png', img)
cv.imwrite(OUTPUT_PATH + "\\" + 'corrected_2D.png', corrected_img)


################################## ETAPA 2. INCERCARE 3D ########################################

INPUT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Gibs_artifact_test\BraTS2021_00000_t2.nii"
OUTPUT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Gibs_artifact_test"

####
# Incaracre poze
####

scan = nib.load(INPUT_PATH).get_fdata()

img = scan[:,:,25]
#print(np.max(img))

img2 = np.uint8(255 * img / np.max(img))
cv.imwrite(OUTPUT_PATH + "\\" + 'original_3D.png', img2)

####
# artefact removal
####

corrected_scan = gibbs_removal_3D(scan)
corrected_img = corrected_scan[:,:,25]

####
# Scalare
####

corrected_img = np.uint8(255 * corrected_img / np.max(corrected_img))

####
# Salvare poze
####

cv.imwrite(OUTPUT_PATH + "\\" + 'corrected_3D.png', corrected_img)












