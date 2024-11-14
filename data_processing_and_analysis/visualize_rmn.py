
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import keyboard

######################## V1) cu nilabel ######################
'''
def wait_till_a_pressed():
    while True:
        if keyboard.is_pressed("a"):
            break

#
scan = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_flair.nii').get_fdata()
#
#scan = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1.nii').get_fdata()
#
#scan = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t1ce.nii').get_fdata()
#
#scan = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_brain_t2.nii').get_fdata()
#
#scan = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000119_final_seg.nii').get_fdata()



#print(scan.shape)

i = 0
test = scan[:,:,i]
plt.imshow(test)
plt.show()

while True:
    i += 5
    wait_till_a_pressed()
    plt.close()
    test = scan[:, :, i]
    plt.imshow(test)
    plt.show()
    #print(i)
'''

######################## V2) cu nilearn ######################
'''
import nilearn
from nilearn.plotting import plot_anat, show

nilearn.plotting.plot_glass_brain(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\00000057_brain_flair.nii')
nilearn.plotting.show()
'''

######################## V3) cu nilabel ######################

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

#
scan1 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_t1.nii').get_fdata()
# FLAIR = Fluid-Attenuated Inversion Recovery
scan2 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_boolseg.nii').get_fdata()
#
scan3 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_t2.nii').get_fdata()
#
scan4 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_t1ce.nii').get_fdata()
# Ground Truth
gt = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00000\BraTS2021_00000_seg.nii').get_fdata()
# BINARY GT - !!!! JUST IF IT EXISTS !!!!
#scan2 = nib.load(r'E:\an_4_LICENTA\Workspace\Dataset\Experiment\BraTS2021_00060\BraTS2021_00060_boolseg.nii').get_fdata()


#print(scan1.shape)

i = 0
test = gt[:,:,i]
plt.imshow(test)
plt.show()

while True:
    wait_till_a_pressed()
    plt.close()
    list_images = [scan1[:,:,i], scan2[:,:,i], scan3[:,:,i], scan4[:,:,i], gt[:,:,i]]

    # #print number of aparitions o  avalue in gt
    unique_gt, counts_gt = np.unique(gt[:,:,i], return_counts=True)
    d = dict(zip(unique_gt, counts_gt))
    #print('GT-ul are urmatoarele valori: ')
    #print(d)

    unique_scan1, counts_scan1 = np.unique(scan1[:, :, i], return_counts=True)
    d = dict(zip(unique_scan1, counts_scan1))
    #print('scan1-ul are urmatoarele valori: ')
    #print(d)

    show_image_list(list_images, figsize=(10, 10))
    #print(i)
    i += 1


