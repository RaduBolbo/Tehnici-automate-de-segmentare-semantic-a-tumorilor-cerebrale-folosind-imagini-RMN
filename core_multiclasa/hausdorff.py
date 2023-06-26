"""
MARE ATENTIE:

1. e obligatoriu sa se introduca data de intrare BINARIZATA !!!!

2. E o problema: ce distanta trebuie sa folosesc? Euclidiana, Manhatten, Daniels, Maurer?

"""

import numpy as np
import cv2 as cv
import SimpleITK as sitk
import matplotlib.pyplot as plt

####
# definire functie
####

def compute_hausdorff_distance(gt, output, label_value, verboise):
    """
    ESENTIAL:
    Trebuie ca output sa fie BINARIZATA !!!!

    INPUT: ndarrays
    """

    #in_img = sitk.ReadImage('seg_contour.nii.gz')
    #in_img = sitk.GetArrayFromImage(in_img)
    #ref_img = sitk.ReadImage('ref_contour.nii.gz')
    #ref_img = sitk.GetArrayFromImage(ref_img)

    gt1 = np.zeros(gt.shape)
    output1 = np.zeros(output.shape)

    # fac o masca binmara pentru acest label (clasa)
    gt1[gt==label_value] = 1.0
    output1[output==label_value] = 1.0

    # Extrag DOAR conturul clasei segmentate !
    # se transforma in imagini de tip SITK pentru a putea aplica functii din biblioteca SimpleITK
    gt1 = sitk.GetImageFromArray(gt1)
    output1 = sitk.GetImageFromArray(output1)

    if verboise:
        ar = sitk.GetArrayFromImage(output1)
        #print(np.max(ar))
        plt.figure(),
        plt.imshow(ar, cmap="gray"), plt.colorbar(), plt.show()

    # asigur precizia
    reference_segmentation = sitk.Cast(gt1, sitk.sitkUInt32)
    seg = sitk.Cast(output1, sitk.sitkUInt32)

    reference_surface = sitk.LabelContour(reference_segmentation, False)
    seg_surface = sitk.LabelContour(seg, False)

    if verboise:
        ar = sitk.GetArrayFromImage(reference_surface)
        ar = ar[:,:,0]
        plt.figure(),
        plt.imshow(ar, cmap = "gray"), plt.colorbar(), plt.show()

    if verboise:
        ar = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(seg_surface, squaredDistance=False, useImageSpacing=True)))
        #print(np.max(ar))
        plt.figure(),
        plt.imshow(ar, cmap="gray"), plt.colorbar(), plt.show()

    # 1. Se vor genera hartile de distante corespunzatoare fiecarui contur. Fiecare pixel din aceste harti de distanta are o valoare egala cu distanta de la el la cel mai apropiat punct al imaginii.
    # 2. Folosind hartile de distante, se determina valorile distantelor de la un set la altul.
    # 3. ATENTIE ! : distantele Maurer si Danielsson este de fapt un mod de laculare a hartilor de distanta Euclidiene
    #seg_distance_map = sitk.Abs(sitk.DanielssonDistanceMap(seg_surface, squaredDistance=False, useImageSpacing=True))
    seg_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg_surface, squaredDistance=False, useImageSpacing=True))
    #reference_segmentation_distance_map = sitk.Abs(sitk.DanielssonDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_segmentation_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))

    if verboise:
        ar = sitk.GetArrayFromImage(seg_distance_map)
        ar = np.clip(np.uint8(ar), 0, 255)
        #print(ar.dtype)
        #print(np.max(ar))
        cv.imshow('pz', ar)
        cv.waitKey()
        cv.destroyAllWindows()

    # Se calculeaza distantele    
    dist_seg = sitk.GetArrayViewFromImage(seg_distance_map)[sitk.GetArrayViewFromImage(reference_surface)==1]
    dist_ref = sitk.GetArrayViewFromImage(reference_segmentation_distance_map)[sitk.GetArrayViewFromImage(seg_surface)==1]

    # La final fac media dintre distante la cuantila 0.95  
    ##print(dist_ref.shape[0], dist_seg.shape[0])
    if dist_ref.shape[0] == 0 or dist_seg.shape[0] == 0:
        return 0
    hausdorf_distance_95 = (np.percentile(dist_ref, 95) + np.percentile(dist_seg, 95)) / 2.0

    return hausdorf_distance_95



def test():
    GT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Hausdorff_test\boolseg_gt.png"
    OUTPUT_PATH = r"E:\an_4_LICENTA\Workspace\Dataset\Hausdorff_test\boolseg_out.png"
    ####
    # Incaracre poze
    ####

    gt = cv.imread(GT_PATH)
    output = cv.imread(OUTPUT_PATH)

    ####
    # artefact removal
    ####

    hausdorff_distance = compute_hausdorff_distance(gt, output, 255, verboise = True)

    ####
    # Afisare rezultat
    ####

    #print(hausdorff_distance)

if __name__ == "__main__":
    test()



















