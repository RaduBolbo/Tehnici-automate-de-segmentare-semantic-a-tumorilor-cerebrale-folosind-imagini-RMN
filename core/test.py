
import cv2

# pentru optimizatori
import torch.optim as optim
import torchvision

# pentru IMPORT-UL DATABASE si transformatele care i se aplica
#from termios import PARODD
import torch
import torchvision  # contine data loader-uri pentru seturi de date comune

# pentru reprezenatare
import matplotlib.pyplot as plt
import numpy as np

# pentru definirea neural network
import torch.nn as nn  # tipuri de straturi
import torch.nn.functional as F  # functii de activare

# pentru optimizatori
import torch.optim as optim

import torchgeometry

from metrici import dice_loss
from metrici import dice_loss_tresholded

from metrici import *

#def test_network(net, test_generator, device, classes, class_acc=True):
def test_network(net, device, test_generator, verboise):
    # se trece modelul in modul de evaluare
    net.eval()

    ################ PE CIFRE ###################
    running_dice_score = 0
    running_dice_score_tresholded = 0
    no_examples = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_generator:
            images, labels = data

            ########
            # CE TINE DE CUDA
            ########
            
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = net(images)
            # #print(outputs.shape)

            ############ Vizualizare_outputs
            if verboise == True:
                # se vor afisa predictia si label-ul. cate onimagine per batch
                #print(outputs.shape)

                img_visaulised = (outputs.numpy())[0 ,0, :,:]
                label_visaulised = (labels.numpy())[0, 0, :, :]
                #print(outputs.shape)
                #print(img_visaulised)
                cv2.imshow('output', img_visaulised)
                cv2.imshow('label', label_visaulised)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
                a = input() # wait to continue
            ############ Vizualizare_outputs

            #L_GDL = diceloss()
            #L_DL = dicescore()
            #L_DL =  torchgeometry.losses.dice_loss
            L_DL = dice_loss
            dice_score = 1 - L_DL(outputs[:,0,:,:], labels[:,0,:,:].type(torch.int64))
            running_dice_score += dice_score

            L_DL = dice_loss_tresholded
            dice_score_tresholded = 1 - L_DL(outputs[:,0,:,:], labels[:,0,:,:].type(torch.int64))
            running_dice_score_tresholded += dice_score_tresholded

            no_examples += 1

        mean_dice_score = running_dice_score/no_examples
        print(f'validation dice_score : {mean_dice_score}')
        mean_dice_score_tresholded = running_dice_score_tresholded/no_examples
        print(f'validation dice_score_tresholded : {mean_dice_score_tresholded}')


        # se trece din nou modelul in modul de antrenare
        net.train()
        return mean_dice_score, mean_dice_score_tresholded # urmeaza sa fac si Hausdorff-Pompeiu



