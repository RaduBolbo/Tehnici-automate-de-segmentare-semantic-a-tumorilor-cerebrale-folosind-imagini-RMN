
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

from metrici_semantic import dice_loss
from metrici_semantic import dice_loss_tresholded

from metrici_semantic import *

#from regula_decizie import apply_regula_decizie_NCR_ET_ED

#from ..hausdorff import compute_hausdorff_distance
from hausdorff import compute_hausdorff_distance

def show_tensor(tensor):
    print(tensor.shape)
    img_numpy = tensor.cpu().numpy()[0,:,:]
    plt.imshow(img_numpy, cmap='gray')
    plt.show()

def specificity_sensitivity(gt, pred, batch_size):
    tp = torch.zeros(batch_size)
    fp = torch.zeros(batch_size)
    tn = torch.zeros(batch_size)
    fn = torch.zeros(batch_size)

    for i in range(batch_size):
        tp[i] = ((gt[i,:,:] == 1) & (pred[i,:,:] == 1)).sum().float()
        fp[i] = ((gt[i,:,:] == 0) & (pred[i,:,:] == 1)).sum().float()
        tn[i] = ((gt[i,:,:] == 0) & (pred[i,:,:] == 0)).sum().float()
        fn[i] = ((gt[i,:,:] == 1) & (pred[i,:,:] == 0)).sum().float()

    specificity = tn / (tn + fp + 0.000001)
    sensitivity = tp / (tp + fn + 0.000001)

    return specificity.mean(), sensitivity.mean()


def compute_housdorff95_on_tensor(gt, pred, batch_size):

    sum_housdorff95 = 0
    for i in range(batch_size):
        #sum_housdorff95 += compute_hausdorff_distance(gt.detach().numpy()[i,:,:], pred[i,:,:].detach().numpy()[i,:,:], 1, verboise = False)
        sum_housdorff95 += compute_hausdorff_distance(gt.cpu().data.numpy()[i,:,:], pred.cpu().data.numpy()[i,:,:], 1, verboise = False)

    return sum_housdorff95/batch_size


#def test_network(net, test_generator, device, classes, class_acc=True):
def test_network(net, device, test_generator, verboise, request_Hausdorff_senz_spec=False):
    # se trece modelul in modul de evaluare
    net.eval()

    ################ PE CIFRE ###################
    running_dice_score_ET = 0
    running_dice_score_ED = 0
    running_dice_score_NCR = 0
    running_dice_score_decided_WT = 0
    running_dice_score_decided_TC = 0
    running_dice_score_decided_ET = 0
    running_senz_WT = 0
    running_senz_TC = 0
    running_senz_ET = 0
    running_spec_WT = 0
    running_spec_TC = 0
    running_spec_ET = 0
    running_hausdorff95_WT = 0
    running_hausdorff95_TC = 0
    running_hausdorff95_ET = 0
    no_examples = 0
    # nu se antreneaza => nu e nevoie de gradienti
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

                img_visaulised = (outputs.numpy())[0 ,0, :, :]
                label_visaulised = (labels.numpy())[0, 0, :, :]
                #print(outputs.shape)
                #print(img_visaulised)
                cv2.imshow('output', img_visaulised)
                cv2.imshow('label', label_visaulised)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
                a = input() # wait to continue
            ############ Vizualizare_outputs

            # VARIANTA 2

            # Extrag GT-urile celor 3 canale:
            target_ET = torch.where(labels[:, 0, :, :] == 255, 1, 0) # 4 este label-ul ET
            target_NCR = torch.where(labels[:, 0, :, :] == 63, 1, 0)
            target_ED = torch.where(labels[:, 0, :, :] == 127, 1, 0) # toate label-urile nenume reprezinta WT

            ##print(torch.unique(labels))
            ##print(torch.unique(target_ET))
            ##print(torch.unique(target_NCR))
            ##print(torch.unique(target_ED))


            # Creez hartile asociate OUTPUT-ului:
            predicted_NCR = outputs[:,0,:,:]
            predicted_ET = outputs[:,1,:,:]
            predicted_ED = outputs[:,2,:,:]

            ##print(torch.unique(predicted_NCR))
            ##print(torch.unique(predicted_ET))
            ##print(torch.unique(predicted_ED))

            ##print(target_ET.shape, predicted_ET.shape)

            # SALV&VIEW $$$$$$$$$
            '''
            target1 = (target_ET.cpu().data.numpy())[0, 0, :, :]
            target2 = (target_NCR.cpu().data.numpy())[0, 0, :, :]
            target3 = (target_ED.cpu().data.numpy())[0, 0, :, :]

            out1 = (predicted_NCR.cpu().data.numpy())[0, :, :]
            out2 = (predicted_ET.cpu().data.numpy())[0, :, :]
            out3 = (predicted_ED.cpu().data.numpy())[0, :, :]

            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target1.png', np.uint8(255*target1))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target2.png', np.uint8(255*target2))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target3.png', np.uint8(255*target3))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\out1.png', np.uint8(255*out1))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\out2.png', np.uint8(255*out2))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\out3.png', np.uint8(255*out3))

            arunc = input('key: ')
            '''
            # SALV&VIEW $$$$$$$$$

            L_DL = dice_loss
            #dice_score_ET = 1 - L_DL(predicted_ET, target_ET.type(torch.int64))
            dice_score_ET = 1 - L_DL(predicted_ET, target_ET.type(torch.int64))
            running_dice_score_ET += dice_score_ET
            #dice_score_NCR = 1 - L_DL(predicted_NCR, target_NCR.type(torch.int64))
            dice_score_NCR = 1 - L_DL(predicted_NCR, target_NCR.type(torch.int64))
            running_dice_score_NCR += dice_score_NCR
            #dice_score_ED = 1 - L_DL(predicted_ED, target_ED.type(torch.int64))
            dice_score_ED = 1 - L_DL(predicted_ED, target_ED.type(torch.int64))
            running_dice_score_ED += dice_score_ED

            # scorurile cerute de ei, practic, DUPA DECIZIE

            ####
            # Calculare GT clase de interes
            ####
            '''
            target_TC = torch.logical_or(target_NCR, target_ET)
            target_WT = torch.logical_or(target_TC, target_ED)
            '''
            # cred ca trebuiau trecute la boolean, caci torch.logical_and nu opereaza pem intregi. BA da, vad ca merge si asa
            target_TC = torch.logical_or(target_NCR.to(torch.bool), target_ET.to(torch.bool))
            target_WT = torch.logical_or(target_TC.to(torch.bool), target_ED.to(torch.bool))
            # conversia de la boolean la 1/0.
            target_TC = torch.where(target_TC[:,:] == True, 1, 0)
            target_WT = torch.where(target_WT[:,:] == True, 1, 0)

            #show_tensor(target_WT)
            ####
            # Luarea DECIZIEI, pe baza iesirii modelului
            ####
            # VARIANTA 1
            '''
            nr_batches = outputs.shape[0]
            decizie = np.zeros((outputs.shape[0], outputs.shape[2], outputs.shape[3]))
            for i in range(nr_batches):
                decizie[i, :, :] = apply_regula_decizie_NCR_ET_ED(outputs.cpu().data.numpy()[i,:,:,:])
            '''

            # VARIANTA 2
            #traduced_output = torch.where((predicted_NCR[:,:] > 0.5) and (predicted_NCR[:,:] > predicted_ET[:,:]) and (predicted_NCR[:,:] > predicted_ED[:,:]), 63, traduced_output[:,:])
            #traduced_output = torch.where(predicted_ED[:,:] > 0.5 and predicted_ED[:,:] > predicted_ET[:,:] and predicted_ED[:,:] > predicted_NCR[:,:], 127, traduced_output[:,:])
            #traduced_output = torch.where(predicted_ET[:,:] > 0.5 and predicted_ET[:,:] > predicted_ED[:,:] and predicted_ET[:,:] > predicted_NCR[:,:], 255, traduced_output[:,:])

            NCR_gt_ET_map = torch.gt(predicted_NCR, predicted_ET)
            NCR_gt_ED_map = torch.gt(predicted_NCR, predicted_ED)
            decided_NCR = torch.logical_and(NCR_gt_ET_map, NCR_gt_ED_map)
            decided_NCR = torch.where(predicted_NCR[:,:] > 0.5, decided_NCR, False) # se verifica daca se trece de 0.5

            ED_gt_ET_map = torch.gt(predicted_ED, predicted_ET)
            ED_gt_NCR_map = torch.gt(predicted_ED, predicted_NCR)
            decided_ED = torch.logical_and(ED_gt_ET_map, ED_gt_NCR_map)
            decided_ED = torch.where(predicted_ED[:,:] > 0.5, decided_ED, False) # se verifica daca se trece de 0.5

            ET_gt_ED_map = torch.gt(predicted_ET, predicted_ED)
            ET_gt_NCR_map = torch.gt(predicted_ET, predicted_NCR)
            decided_ET = torch.logical_and(ET_gt_ED_map, ET_gt_NCR_map)
            decided_ET = torch.where(predicted_ET[:,:] > 0.5, decided_ET, False) # se verifica daca se trece de 0.5


            '''
            decizie = torch.tensor(decizie)
            decided_ET = torch.where(decizie[:, :, :] == 255, 1, 0) # 4 este label-ul ET
            decided_NCR = torch.where(decizie[:, :, :] == 63, 1, 0)
            decided_ED = torch.where(decizie[:, :, :] == 127, 1, 0) # toate label-urile nenume reprezinta WT
            '''

            ####
            # Calcularea predictiilor pe baza DECIZIEI
            ####
            ##print('lllllllllllllllllll')
            ##print(torch.unique(decided_NCR))
            ##print(torch.unique(decided_ED))
            ##print(torch.unique(decided_ET))
            '''
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\labels!.png', np.uint8(labels.cpu().data.numpy()[0,0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\decided_NCR.png', np.uint8(255*decided_NCR.cpu().data.numpy()[0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\decided_ET.png', np.uint8(255*decided_ET.cpu().data.numpy()[0,:,:]))
            '''

            
            decided_TC = torch.logical_or(decided_NCR, decided_ET) # TC = NCR + ET
            decided_WT = torch.logical_or(decided_TC, decided_ED) # WT = TC + ED
            # conversia de la boolean la 1/0.
            decided_TC = torch.where(decided_TC[:,:] == True, 1, 0)
            decided_WT = torch.where(decided_WT[:,:] == True, 1, 0)
            decided_ET = torch.where(decided_ET[:,:] == True, 1, 0)
            '''
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\decided_TC.png', np.uint8(255*decided_TC.cpu().data.numpy()[0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\decided_WT.png', np.uint8(255*decided_WT.cpu().data.numpy()[0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target_WT.png', np.uint8(255*target_WT.cpu().data.numpy()[0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target_TC.png', np.uint8(255*target_TC.cpu().data.numpy()[0,:,:]))
            cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\target_ET.png', np.uint8(255*target_ET.cpu().data.numpy()[0,:,:]))
            a = input()
            '''

            ##print(torch.unique(decided_TC))
            ##print(torch.unique(decided_WT))

            ####
            # Calcularea DICE SCORE pt clasele de interes
            ####
            dice_score_decided_WT = 1 - L_DL(decided_WT, target_WT.type(torch.int64).to(device='cuda:0'))
            running_dice_score_decided_WT += dice_score_decided_WT
            
            dice_score_decided_TC = 1 - L_DL(decided_TC, target_TC.type(torch.int64).to(device='cuda:0'))
            running_dice_score_decided_TC += dice_score_decided_TC
            
            dice_score_decided_ET = 1 - L_DL(decided_ET, target_ET.type(torch.int64).to(device='cuda:0'))
            running_dice_score_decided_ET += dice_score_decided_ET

            '''
            L_DL = dice_loss_tresholded
            dice_score_tresholded = 1 - L_DL(outputs[:,0,:,:], labels[:,0,:,:].type(torch.int64))
            running_dice_score_tresholded += dice_score_tresholded
            '''

            ########
            # PARTEA a 2-a: se cacluleaza Hausdorff95, senzitivitate, specificitate
            ########

            if request_Hausdorff_senz_spec:
                ####
                # Senzitivitate & Specificitate
                ####
                # WT
                act_spec_WT, act_snez_WT = specificity_sensitivity(target_WT, decided_WT, labels.shape[0])
                running_spec_WT += act_spec_WT
                running_senz_WT += act_snez_WT
                # TC
                act_spec_TC, act_snez_TC = specificity_sensitivity(target_TC, decided_TC, labels.shape[0])
                running_spec_TC += act_spec_TC
                running_senz_TC += act_snez_TC
                # ET
                act_spec_ET, act_snez_ET = specificity_sensitivity(target_ET, decided_ET, labels.shape[0])
                running_spec_ET += act_spec_ET
                running_senz_ET += act_snez_ET
                ####
                # Hausdorff95
                ####
                running_hausdorff95_WT += compute_housdorff95_on_tensor(target_WT, decided_WT, labels.shape[0])
                running_hausdorff95_TC += compute_housdorff95_on_tensor(target_TC, decided_TC, labels.shape[0])
                running_hausdorff95_ET += compute_housdorff95_on_tensor(target_ET, decided_ET, labels.shape[0])
                

            no_examples += 1

        # Partea de dafoiisare

        # 1) SCORURILE ET ED NCR
        mean_dice_score_ET = running_dice_score_ET/no_examples
        mean_dice_score_NCR = running_dice_score_NCR/no_examples
        mean_dice_score_ED = running_dice_score_ED/no_examples
        print(f'val DS_ET : {mean_dice_score_ET}')
        print(f'val DS_NCR : {mean_dice_score_NCR}')
        print(f'val DS_ED : {mean_dice_score_ED}')
        
        # 2) SCORURILE SOLICITATE: 
        mean_decided_dice_score_WT = running_dice_score_decided_WT/no_examples
        mean_decided_dice_score_TC = running_dice_score_decided_TC/no_examples
        mean_decided_dice_score_ET = running_dice_score_decided_ET/no_examples
        print(f'val decided DS_WT : {mean_decided_dice_score_WT}')
        print(f'val decided DS_TC : {mean_decided_dice_score_TC}')
        print(f'val decided DS_ET : {mean_decided_dice_score_ET}')

        ########
        # PARTEA a 2-a: se afiseaza Hausdorff95, senzitivitate, specificitate
        ########
        if request_Hausdorff_senz_spec:
            ####
            # Senzitivitate & Specificitate
            ####
            mean_senz_WT = running_senz_WT/no_examples
            mean_spec_WT = running_spec_WT/no_examples
            mean_senz_TC = running_senz_TC/no_examples
            mean_spec_TC = running_spec_TC/no_examples
            mean_senz_ET = running_senz_ET/no_examples
            mean_spec_ET = running_spec_ET/no_examples
            print(f'val decided senz_WT : {mean_senz_WT}')
            print(f'val decided senz_TC : {mean_senz_TC}')
            print(f'val decided senz_ET : {mean_senz_ET}')
            print(f'val decided spec_WT : {mean_spec_WT}')
            print(f'val decided spec_TC : {mean_spec_TC}')
            print(f'val decided spec_ET : {mean_spec_ET}')
            ####
            # Hausdorff95
            ####
            hausdorff95_WT = running_hausdorff95_WT/no_examples
            hausdorff95_TC = running_hausdorff95_TC/no_examples
            hausdorff95_ET = running_hausdorff95_ET/no_examples
            print(f'val hausdorff95_WT : {hausdorff95_WT}')
            print(f'val hausdorff95_TC : {hausdorff95_TC}')
            print(f'val hausdorff95_ET : {hausdorff95_ET}')
            



        # se trece din nou modelul in modul de antrenare
        net.train()
        if request_Hausdorff_senz_spec:
            return mean_dice_score_ET, mean_dice_score_NCR, mean_dice_score_ED, mean_decided_dice_score_WT, mean_decided_dice_score_TC, mean_decided_dice_score_ET, mean_senz_WT, mean_senz_TC, mean_senz_ET, mean_spec_WT, mean_spec_TC, mean_spec_ET, hausdorff95_WT, hausdorff95_TC, hausdorff95_ET # urmeaza sa fac si Hausdorff-Pompeiu
        else:
            return mean_dice_score_ET, mean_dice_score_NCR, mean_dice_score_ED, mean_decided_dice_score_WT, mean_decided_dice_score_TC, mean_decided_dice_score_ET # urmeaza sa fac si Hausdorff-Pompeiu



