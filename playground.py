
import nibabel as nib
import numpy as np
import socket
import matplotlib.pyplot as plt
import keyboard
import SimpleITK as sitk
from skimage import exposure
from nipype.interfaces.ants import N4BiasFieldCorrection
import skimage
import os
from PIL import Image

#pentru micul lot de optimizare 
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_flair.nii'
OUTPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_flair_out.nii'

INITIAL = r"E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000"
FINAL = r"E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000"

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


def n4_bias_correction(image_path, image_path_out):
    '''
        PARTAMETRII MENIONATI IN APPERUL:
        https://bmjopen.bmj.com/content/bmjopen/12/7/e059000.full.pdf
    '''
    ##print(type(image))
    n4 = N4BiasFieldCorrection()
    ##print(dir(n4))
    n4.inputs.dimension = 3
    n4.inputs.input_image = image_path
    n4.inputs.bspline_fitting_distance = 100 # * dar investighez valoare lui SetNumberOfControlPoint pt 4, 3, 2
    n4.inputs.shrink_factor = 2
    n4.inputs.convergence_threshold = 0 # *
    n4.inputs.n_iterations = [20,20,20,10] # *?

    n4.inputs.output_image = image_path_out
    #n4.inputs.save_bias = True

    #n4.cmdline
    res = n4.run()


    

    return res.outputs

# def n4_bias_correction(image_path, image_path_out):
#     '''
#         PARTAMETRII MENIONATI IN APPERUL:
#         https://bmjopen.bmj.com/content/bmjopen/12/7/e059000.full.pdf
#     '''
#     ##print(type(image))
#     n4 = N4BiasFieldCorrection()
#     ##print(dir(n4))
#     n4.inputs.dimension = 3
#     n4.inputs.input_image = image_path
#     n4.inputs.bspline_fitting_distance = 100
#     n4.inputs.shrink_factor = 2
#     n4.inputs.convergence_threshold = 0
#     n4.inputs.n_iterations = [20,20,20,10]

#     n4.inputs.output_image = image_path_out
#     #n4.inputs.save_bias = True

#     n4.cmdline

def corectie_bias_slice(img):
    normalized_image = min_max_normalization(img)
    bias_corrected = n4_bias_correction(normalized_image)
    bias_corrected_restored = extend_to_0_255(bias_corrected)
    return bias_corrected_restored



scan_t1_loadfull = nib.load(INPUT_PATH, mmap=False)
scan_t1 = scan_t1_loadfull.get_fdata()
img1 = Image.fromarray(np.uint8(255*scan_t1[:,:,25]/np.max(scan_t1[:,:,25])))
#img = img.convert('L')
img1.save(INITIAL + '\\' + 'flairinitial.png')

output = n4_bias_correction(INPUT_PATH, OUTPUT_PATH)
#print(output)

scan_t2_loadfull = nib.load(OUTPUT_PATH, mmap=False)
scan_t2 = scan_t1_loadfull.get_fdata()
img2 = Image.fromarray(np.uint8(255*scan_t2[:,:,25]/np.max(scan_t2[:,:,25])))
#img = img.convert('L')
img2.save(FINAL + '\\' + 'flairfinal.png')

dif = np.abs(np.array(img1)-np.array(img2))
dif = Image.fromarray(np.uint8(255*dif/np.max(dif)))
#dif = Image.fromarray(np.uint8(dif))
dif.save(INITIAL + '\\' + 'dif.png')