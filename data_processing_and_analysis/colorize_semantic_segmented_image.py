
import numpy as np
import cv2 as cv
import os


def colorize_path(IMG_PATH):
    '''
    IMG_PATH = path of the colorized image
    B Blue = NCR (63)
    G Green = ET (255)
    R Red = ED (127)
    '''

    img = cv.imread(IMG_PATH)
    h, w, c = img.shape
    colorized_img = np.zeros((h, w, 3))
    colorized_img[:,:,0] = np.where(img[:,:,0] == 63, 255, 0) # B = NCR
    colorized_img[:,:,1] = np.where(img[:,:,0] == 255, 255, 0) # G = ED
    colorized_img[:,:,2] = np.where(img[:,:,0] == 127, 255, 0) # R = ET
    cv.imwrite(IMG_PATH[:-4] + '_color.png', colorized_img)


def colorize_recurs(DIR_PATH, IMG_NAME):
    '''
    This method colorizes each image with the name 'IMG_NAME', under the directory 'DIR_PATH'
    '''
    for DIR2 in os.listdir(DIR_PATH):
        for filename in os.listdir(DIR_PATH + '\\' + DIR2):
            filepath = DIR_PATH + '\\' + DIR2 + '\\' + filename
            
            if filename == IMG_NAME:
                colorize_path(filepath)

def colorize_all_imgs_in_dir(DIR_PATH):
    '''
    This method colorizes each image with the name 'IMG_NAME', under the directory 'DIR_PATH'
    '''
    for filename in os.listdir(DIR_PATH):
        filepath = DIR_PATH + '\\' + filename
        colorize_path(filepath)

# 1) Colorize image

'''
IMG_PATH = r'E:\an_4_LICENTA\Workspace\junkdata\to_colorize.png'
colorize_path(IMG_PATH)
'''

# 2) Colorize RECURSIVELY a whole directory

'''
DIR_PATH = r'E:\an_4_LICENTA\Workspace\inference_workspace\input'
colorize_recurs(DIR_PATH, 'seg.png')
'''

# 3) Colorize all images in a dir3ectory
DIR_PATH = r'E:\an_4_LICENTA\Workspace\inference_workspace\output_semantic'
colorize_all_imgs_in_dir(DIR_PATH)


