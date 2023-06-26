
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import SimpleITK as sitk
from skimage import exposure
import skimage

def wait_till_a_pressed():
    while True:
        if keyboard.is_pressed("a"):
            break

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False



def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30):

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


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
    #print(min, max)
    rescaled_image = (img - min)*256 / (max - min)
    return np.uint8(rescaled_image)

def n4_bias_correction(image):
    ##print(type(image))
    image = sitk.Cast(image, sitk.sitkFloat32)
    #image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_corrected = corrector.Execute(image)
    return output_corrected

################# CORECTIE 2D #######################

#
#scan1 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_flair.nii').get_fdata()
#scan1 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00011\BraTS2021_00011_flair.nii').get_fdata()
scan1 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00020\BraTS2021_00020_flair.nii').get_fdata()
#
#scan2 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1.nii').get_fdata()
#
#scan3 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1ce.nii').get_fdata()
#
#scan4 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t2.nii').get_fdata()
# Ground Truth
#gt = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_final_seg.nii').get_fdata()



#print(scan1.shape)

test_1 = scan1
#test_1 = scan1[:,:,25]
#test_2 = scan2[:,:,70]
#test_3 = scan3[:,:,70]
#test_4 = scan4[:,:,70]
#plt.imshow(test_1)
#plt.show()

# plotare histograme

m_normalized_image = min_max_normalization(test_1)
m_array = sitk.GetArrayFromImage(m_normalized_image)
hist, hist_centers = exposure.histogram(m_array, nbins = 256)
imag_cdf, bins = exposure.cumulative_distribution(m_array)


plt.plot(bins, imag_cdf, 'r')
plt.plot(hist_centers, hist, 'b')
plt.show()

# APLICARE CORECTIE:

#! normarea se face INAINTE de n4_bias_correction
m_normalized_image_2 = min_max_normalization(test_1)
bias_corrected_test_1 = n4_bias_correction(m_normalized_image_2)

m_array = sitk.GetArrayFromImage(bias_corrected_test_1)
hist, hist_centers = exposure.histogram(m_array, nbins = 256)
imag_cdf, bins = exposure.cumulative_distribution(m_array)

plt.plot(bins, imag_cdf, 'r')
plt.plot(hist_centers, hist, 'b')
plt.show()

# Afisarea diferentei dintre imaginea cu bias field si ce acu corecatat
#print('lllll')
#print(np.min(m_normalized_image), np.max(m_normalized_image))
#print(np.min(m_normalized_image[25,:,:]), np.max(m_normalized_image[25,:,:]))

test_1_inital_restored = extend_to_0_255(m_normalized_image[25,:,:])
plt.imshow(test_1_inital_restored)
plt.show()
test_1_corrected_restored = extend_to_0_255(bias_corrected_test_1[25,:,:])
plt.imshow(test_1_corrected_restored)
plt.show()
dif = test_1_inital_restored - test_1_corrected_restored

plt.imshow(dif)
plt.show()




################# CORECTIE 2D #######################
'''

#
scan1 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_flair.nii').get_fdata()
#
#scan2 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1.nii').get_fdata()
#
#scan3 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1ce.nii').get_fdata()
#
#scan4 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t2.nii').get_fdata()
# Ground Truth
#gt = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_final_seg.nii').get_fdata()



#print(scan1.shape)

test_1 = scan1[:,:,25]
#test_2 = scan2[:,:,70]
#test_3 = scan3[:,:,70]
#test_4 = scan4[:,:,70]
plt.imshow(test_1)
plt.show()

# plotare histograme

m_normalized_image = min_max_normalization(test_1)
m_array = sitk.GetArrayFromImage(m_normalized_image)
hist, hist_centers = exposure.histogram(m_array, nbins = 256)
imag_cdf, bins = exposure.cumulative_distribution(m_array)


plt.plot(bins, imag_cdf, 'r')
plt.plot(hist_centers, hist, 'b')
plt.show()

# APLICARE CORECTIE:

#! normarea se face INAINTE de n4_bias_correction
m_normalized_image_2 = min_max_normalization(test_1)
bias_corrected_test_1 = n4_bias_correction(m_normalized_image_2)

m_array = sitk.GetArrayFromImage(bias_corrected_test_1)
hist, hist_centers = exposure.histogram(m_array, nbins = 256)
imag_cdf, bins = exposure.cumulative_distribution(m_array)

plt.plot(bins, imag_cdf, 'r')
plt.plot(hist_centers, hist, 'b')
plt.show()

# Afisarea diferentei dintre imaginea cu bias field si ce acu corecatat

test_1_inital_restored = extend_to_0_255(m_normalized_image)
plt.imshow(test_1_inital_restored)
plt.show()
test_1_corrected_restored = extend_to_0_255(bias_corrected_test_1)
plt.imshow(test_1_corrected_restored)
plt.show()
dif = test_1_inital_restored - test_1_corrected_restored

plt.imshow(dif)
plt.show()

'''
















