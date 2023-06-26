import torch
from test import *

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru optimizatori
import torch.optim as optim

# pentru tensorboard
from torch.utils.tensorboard import SummaryWriter

# metrici
from metrici import dice_loss
import metrici

# pentru dice loss
import torchgeometry

# pt validare
from test import test_network
import os



def save_model(net, optimizer, epoch, loss, current_score, current_best_score, SAVE_PATH, mod):
    if mod == 'complete_save':
        current_best_score = current_best_score
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'current_best_score': current_best_score,
            }, SAVE_PATH + '\\model' + str(epoch) + '.pth')
    else:
            current_best_score = current_score
            torch.save(net.state_dict(), SAVE_PATH + '\\model' + str(epoch) + '.pth')
    return current_best_score

#def train_network(net, device, trainloader, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_acc,mod):
def train_network(net, device, trainloader, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_acc, start_epoch, mod, view_outputs=False):
    ######
    # Ce tine de cuda
    ######
    net.to(device)
    
    # pentru tensorboard, se instantiaza un writer:
    #writer = SummaryWriter(log_dir='runs')
    writer = SummaryWriter(log_dir='runs')
    #summary = sess.run(merged, feed_dict=feed_dict)
    # se foloseste si un fisier:
    #File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_binar_brats_2D_v1_nepreprocesat_64_128_256.txt", "a")
    #File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_binar_brats_2D_MaxIter10_nobins256_CtrlPts6_v3_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bilinear_.txt", "a")
    #File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_Augumentare_FLIP,ZOOM_GAUSS_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bil.txt", "a")
    ##print(os.listdir(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results"))
    #File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\resnet_1_32fx24_corectat.txt", "a")
    #File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\resnet_1_32fx8_64fx8_128fx8_corectat.txt", "a")
    File_object = open(r"E:\an_4_LICENTA\Workspace\Scripturi\core\results\resnet_1_128fx24_corectat.txt", "a")


    # CRITERION = functia de loss.
    #L_CE = nn.BCELoss()
    #criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #criterion = torchgeometry.losses.dice_loss
    criterion = metrici.dice_loss
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    #? aici mai trebuie lucrat. Poate fac un fel de fnctie de loss compusa

    # PENTRU MAI MULTE CLASE
    #criterion = nn.CrossEntropyLoss()
    # PENTRU DOAR 2 CLASE

    # for epoch in tqdm.auto.tqdm(range(nr_epochs)):  # loop over the dataset multiple times
    for epoch in range(start_epoch, nr_epochs):

        running_loss = 0.0
        no_examples = 0
        # for i, data in tqdm.auto.tqdm(enumerate(trainloader)):
        for i, data in enumerate(tqdm(trainloader)):
            image, label = data
            ##print('data shape:' + str(image.shape) + '    ' + str(label.shape))
            # label = label[0]

            ########
            # CE TINE DE CUDA
            ########
            
            image = image.to(device).float()
            label = label.to(device)
            # inputs, labels = data[0].to(device), data[1].to(device)

            # Se seteaza la ZERO gradientii optimkizarii (care erau nenuli de la iteratia precedenta)
            # asta se face pentru ca gradientii sa provina exclusiv din acest backprop si nu sa se compuna cu cei de la iteratia precedenta
            optimizer.zero_grad()
            # #print(inputs.shape)
            outputs = net(image)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net

            ############ Vizualizare_outputs
            #view_outputs = True
            if view_outputs == True:
                # se vor afisa predictia si label-ul. cate onimagine per batch
                #print(outputs.shape)
                img_visaulised = (outputs.detach().numpy())[0, 0, :, :]
                label_visaulised = (label.detach().numpy())[0, 0, :, :]
                #print(outputs.shape)
                #print(img_visaulised)
                cv2.imshow('output', img_visaulised)
                cv2.imshow('label', label_visaulised)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
                a = input()  # wait to continue
            ############ Vizualizare_outputs

            ##### Secventa pt ca nu am cuda #### e inclusa in secventa comenatat pt cyuda de mai sus ####

            ########
            ## Varianta 1 - custrom loss. Nu merge. ideea e ca loss-ul trebuie sa fie DIFERENTIABIL. Cred ca aceasta medoda rupe graful computuaional
            ########
            #loss = criterion_binara(outputs, label, verboise=False)

            ########
            ## Varianta 2 - BCELoss. Asta sigur e diferentiabil
            ########
            
            # MSE
            #loss = criterion(outputs, label)

            # dice_loss !! AM MODIFICAT BIBLIOTECA !!
            #loss = criterion(outputs, label)
            loss = criterion(outputs[:,0,:,:], label[:,0,:,:].type(torch.int64)) # se elimina canalul unic de 'gri' oricum einutil

            loss.backward()  # se calc. VALOAREA NUMERICA loss, aplicand functia de cost CRITERION, labelurilor
            optimizer.step()  # se realizeaza efectiv backprop-ul, iterand prin TOTI TENSORII cu PARAMETRII

            running_loss += loss.item()
            no_examples = i

        # se updateaza optimizatorul, cu scheduler-ul care schimba LR-ul
        scheduler.step()

        # pentru tensorboard
        epoch_loss = running_loss / no_examples
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        print(f'[{epoch + 1}, {i + 1:8d}] loss: {epoch_loss:.8f}')

        # VALIDARE:
        #current_acc = test_network(net, val_generator, device, classes, class_acc=False)
        current_dice_score, current_dice_score_trasholded = test_network(net, device, val_generator, verboise=False)
        writer.add_scalar("Dice_score/val", current_dice_score, epoch) # este dice score
        writer.add_scalar("Dice_score_tresholded/val", current_dice_score_trasholded, epoch) # este dice score
        # scriu rezultatele si in fisier
        File_object.write(f"Epoch {epoch}, Loss: {epoch_loss}, Dice_score: {current_dice_score}, Dice_score_tresholded:{current_dice_score_trasholded} \n")
        File_object.flush()

        # salvare graf computational
        images, labels = next(iter(trainloader))
        images = images.to(device).float()
        labels = labels.to(device)

        #writer.add_graph(net, images)
        # ?????????????????????????????????????????
        writer.flush()
        # ?????????????????????????????????????????

        # salvare model, daca e cel mai bun d eapa acum (!!!! ar trebui sa salvez oricum, mai inol, neconditional\t)
        current_best_score = save_model(net, optimizer, epoch, loss, current_dice_score, current_best_acc, SAVE_PATH, mod)

    #print('Finished Training')

    # se asigura ca toate datele sunt scrise bine pe disk
    writer.flush()
    # se inchide writer-ul. Nu se mai ppoate scrie inn el
    writer.close()
    # se inchide si sisierul
    File_object.close()

    return net

