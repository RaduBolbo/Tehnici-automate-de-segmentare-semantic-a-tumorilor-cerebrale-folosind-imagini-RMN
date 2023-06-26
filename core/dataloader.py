import cv2
import torch
import os
import numpy as np
import imutils
import random
import nibabel as nib

import torchvision.transforms as transforms


def one_hot_encoding(label_list):
    unique, inverse = np.unique(label_list, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return inverse


class Dataset(torch.utils.data.Dataset):

    def __init__(self, DIR_PATH, normalize, zoom=False, rotate=False, horiz_flip=False, vert_flip=False, gauss_disp=0):
        secondary_paths = []
        for dir_name_2 in os.listdir(DIR_PATH):
            DIR_PATH_2 = DIR_PATH + '\\' + dir_name_2
            secondary_paths.append(DIR_PATH_2)
        # se transforma labels, din lista de str, in pne hut

        self.DIR_PATH = DIR_PATH
        self.secondary_paths = secondary_paths
        #self.encoded_labels = one_hot_encoding(self.labels)
        #self.img_path = img_path
        self.normalize = normalize
        self.rotate = rotate
        self.zoom = zoom
        self.horiz_flip = horiz_flip
        self.vert_flip = vert_flip
        self.gauss_disp = gauss_disp
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.14939, 0.07954, 0.08472, 0.10318], std=[0.27481, 0.15351, 0.16901, 0.20120])])

    def __len__(self):
        return len(self.secondary_paths)

    def __getitem__(self, index):
        # ele se vor incarca cu 3 canale, dar valorile pe cele 3 ch sutn indentice
        x_t1 = cv2.imread(self.secondary_paths[index] + '\\' + 't1.png')
        x_t1ce = cv2.imread(self.secondary_paths[index] + '\\' + 't1ce.png')
        x_t2 = cv2.imread(self.secondary_paths[index] + '\\' + 't2.png')
        x_flair = cv2.imread(self.secondary_paths[index] + '\\' + 'flair.png')
        y = cv2.imread(self.secondary_paths[index] + '\\' + 'boolseg.png')

        if self.horiz_flip:
            n = np.random.uniform(0,1)
            if n>0.5:
                x_t1 = cv2.flip(x_t1, 1)
                x_t1ce = cv2.flip(x_t1ce, 1)
                x_t2 = cv2.flip(x_t2, 1)
                x_flair = cv2.flip(x_flair, 1)
                y = cv2.flip(y, 1)

        if self.vert_flip:
            n = np.random.uniform(0,1)
            if n>0.5:
                x_t1 = cv2.flip(x_t1, 0)
                x_t1ce = cv2.flip(x_t1ce, 0)
                x_t2 = cv2.flip(x_t2, 0)
                x_flair = cv2.flip(x_flair, 0)
                y = cv2.flip(y, 0)

        if self.zoom:
            #zoom_factor_x = np.random.exponential(scale=0.1)
            #zoom_factor_y = np.random.exponential(scale=0.1)
            zoom_factor_x = np.random.uniform(0,0.25)
            zoom_factor_y = np.random.uniform(0,0.25)
            if zoom_factor_x < 0.25 and zoom_factor_x > 0.05 and zoom_factor_y < 0.25 and zoom_factor_y > 0.05:
                x_t1 = x_t1[int(np.ceil(x_t1.shape[0] * zoom_factor_x)):x_t1.shape[0],
                    int(np.ceil(x_t1.shape[0] * zoom_factor_y)):x_t1.shape[1], :]
                x_t1 = cv2.resize(x_t1, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_t1ce = x_t1ce[int(np.ceil(x_t1ce.shape[0] * zoom_factor_x)):x_t1ce.shape[0],
                    int(np.ceil(x_t1ce.shape[0] * zoom_factor_y)):x_t1ce.shape[1], :]
                x_t1ce = cv2.resize(x_t1ce, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_t2 = x_t2[int(np.ceil(x_t2.shape[0] * zoom_factor_x)):x_t2.shape[0],
                    int(np.ceil(x_t2.shape[0] * zoom_factor_y)):x_t2.shape[1], :]
                x_t2 = cv2.resize(x_t2, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_flair = x_flair[int(np.ceil(x_flair.shape[0] * zoom_factor_x)):x_flair.shape[0],
                    int(np.ceil(x_flair.shape[0] * zoom_factor_y)):x_flair.shape[1], :]
                x_flair = cv2.resize(x_flair, (240, 240), interpolation = cv2.INTER_LINEAR)
                y = y[int(np.ceil(y.shape[0] * zoom_factor_x)):y.shape[0],
                    int(np.ceil(y.shape[0] * zoom_factor_y)):y.shape[1], :]
                y = cv2.resize(y, (240, 240), interpolation = cv2.INTER_LINEAR)

        if self.rotate:
            # se genereaza unghiul de rotatie, dinbtr-oi distributie exponentiala.
            #rot_angle = int(np.ceil(10 * np.random.exponential(scale=1)))
            # sau se extrage intr-o distributie uniforma
            rot_angle = np.random.uniform(0,90)
            # se plafoneaza unghiul, in cazutile extreme
            if rot_angle > 90:
                rot_angle = 90
            n = np.random.uniform(0,2)
            if n > 1:
                if n>1.5:
                    rot_angle = -rot_angle
                # se roteste imagionea
                # x = imutils.rotate(x, rot_angle)
                x_t1 = imutils.rotate_bound(x_t1, rot_angle)
                x_t1 = cv2.resize(x_t1, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_t1ce = imutils.rotate_bound(x_t1ce, rot_angle)
                x_t1ce = cv2.resize(x_t1ce, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_t2 = imutils.rotate_bound(x_t2, rot_angle)
                x_t2 = cv2.resize(x_t2, (240, 240), interpolation = cv2.INTER_LINEAR)
                x_flair = imutils.rotate_bound(x_flair, rot_angle)
                x_flair = cv2.resize(x_flair, (240, 240), interpolation = cv2.INTER_LINEAR)
                y = imutils.rotate_bound(y, rot_angle)
                y = cv2.resize(y, (240, 240), interpolation = cv2.INTER_LINEAR)

        

        '''
        # VIZAULIZARE
        cv2.imshow('x_t1', x_t1)
        cv2.imshow('x_t1ce', x_t1ce)
        cv2.imshow('x_t2', x_t2)
        cv2.imshow('x_flair', x_flair)
        cv2.imshow('y', y)
        cv2.waitKey()
        '''

        # Normalizare canal cu canal
        #? eok sa fie ch cu ch sau ar trebui sa normlizez tot volumul - sa impart la maximul global?
        #? eu zic ca e mai bine asa
        x_t1 = np.double(x_t1[:,:,0] / np.max(x_t1[:,:,0]))
        x_t1ce = np.double(x_t1ce[:,:,0] / np.max(x_t1ce[:,:,0]))
        x_t2 = np.double(x_t2[:,:,0] / np.max(x_t2[:,:,0]))
        x_flair = np.double(x_flair[:,:,0] / np.max(x_flair[:,:,0]))
        ##print(x_t1.shape)

        y = np.double(y[:,:,0]/ np.max(y[:,:,0])) # si label-ul TREBUIE sa fie intre 0 si 1, ca sa am rezulate corecte la loss

        # agrgarea celor 4 canale
        x = np.dstack((x_t1, x_t1ce, x_t2, x_flair))
        y = np.dstack((y))
        ##print(x.shape)

        if self.gauss_disp != 0:
            h, w, c = x.shape
            gauss_noise = np.random.normal(0, self.gauss_disp, (h, w, c))
            x = x + gauss_noise


        ##print('dstacked shape 1 : ' + str(x.shape))
        

        if self.normalize == '0_mean_1std':
            if self.transform:
                x = self.transform(x)
            x.numpy()
            #x = np.swapaxes(x, 2, 0)
        else:
            # e nevoie, pentru ca in pytorch, se asteapta [batch_size, no_ch, h, w]
            ##print('dstacked shape 1 : ' + str(x.shape))
            x = np.swapaxes(x, 2, 0)
            #x = np.swapaxes(x, 2, 0) # poate asa nu mai intoarec labelurile
            #x = np.swapaxes(x, 1, 2)
            #y = np.swapaxes(x, 1, 2)
            

        ##print('dataloader returned x shape: ' + str(x.shape))
        ##print('dataloader returned y shape: ' + str(y.shape))
        return x, y
    '''
    def afisare(self):
        for index in range(len(self.names)):
            ID = self.names[index]
            #print(self.img_path + self.labels[index] + '/' + ID)
    '''

def load_complete_model(checkpoint, PATH, net, optimizer):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    current_best_acc = checkpoint['current_best_acc']

    return net, optimizer, epoch, loss, current_best_acc




