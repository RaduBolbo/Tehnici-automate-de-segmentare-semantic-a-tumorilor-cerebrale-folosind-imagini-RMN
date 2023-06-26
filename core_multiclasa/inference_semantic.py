
'''

'''


########
# INPORTS
########

from tabnanny import verbose
from dataloader_semantic import *
# from retele import *
from train2_semantic import *

import torch
import numpy as np

torch.manual_seed(0)

# pentru incarcarea retelelor:
from load_checkpoint import load_complete_model

###### import-uri pentru retele
# from networks import *
from networks_folder_semantic.unet_semantic_brats_2D_v3_COPIE import UNET_semantic_brats_2D_v3_COPIE
import tqdm
import cv2

def apply_regula_decizie_NCR_ET_ED(output):
    '''
        Defineste regula dupa care sunt interperetate iesirile retelei

        !! Folosita in cazul in care iesirele sunt, in aceasta ordine: NCR, ET, ED

        REGULA:
        voxcelul se atribuie clasei stratului de activare maxima, daca trece de pragul de 0.5
    '''
    # Creez hartile asociate OUT-ului:
    ##print(output.shape)
    predicted_NCR = output[0,:,:]
    predicted_ET = output[1,:,:]
    predicted_ED = output[2,:,:]

    h, w = predicted_NCR.shape

    traduced_output = np.uint8(np.zeros((h, w)))

    ####
    # VARIANTA 1 - NU MERGE, DI CAUZA "and"
    ####
    '''
    traduced_output = np.where(predicted_NCR[:,:] > 0.5 and predicted_NCR[:,:] > predicted_ET[:,:] and predicted_NCR[:,:] > predicted_ED[:,:], 63, traduced_output[:,:])
    traduced_output = np.where(predicted_ED[:,:] > 0.5 and predicted_ED[:,:] > predicted_ET[:,:] and predicted_ED[:,:] > predicted_NCR[:,:], 127, traduced_output[:,:])
    traduced_output = np.where(predicted_ET[:,:] > 0.5 and predicted_ET[:,:] > predicted_ED[:,:] and predicted_ET[:,:] > predicted_NCR[:,:], 255, traduced_output[:,:])
    '''
    ####
    # VARIANTA 2 - ineficienta
    ####
    #print(np.unique(predicted_ED))
    for i in range(h):
        for j in range(w):
            if predicted_NCR[i,j] > 0.5 and predicted_NCR[i,j] > predicted_ET[i,j] and predicted_NCR[i,j] > predicted_ED[i,j]:
                traduced_output[i, j] = 63
            elif predicted_ED[i,j] > 0.5 and predicted_ED[i,j] > predicted_ET[i,j] and predicted_ED[i,j] > predicted_NCR[i,j]:
                traduced_output[i, j] = 127
            elif predicted_ET[i,j] > 0.5 and predicted_ET[i,j] > predicted_ED[i,j] and predicted_ET[i,j] > predicted_NCR[i,j]:
                traduced_output[i, j] = 255
            elif predicted_ET[i,j] < 0.5 and predicted_ED[i,j] < 0.5 and predicted_NCR[i,j] < 0.5:
                traduced_output[i, j] = 0
    
    return traduced_output



def inference_on_dataset_1doutout_semantic(net, inference_generator, SAVE_PATH):
    net.eval()
    for i, data in enumerate(inference_generator):
        image, label = data
        ########
        # CE TINE DE CUDA
        ########
        
        '''
        inputs = image.to(device).float()
        labels = label.to(device)
        # inputs, labels = data[0].to(device), data[1].to(device)

        # Se seteaza la ZERO gradientii optimkizarii (care erau nenuli de la iteratia precedenta)
        optimizer.zero_grad()
        # #print(inputs.shape)
        outputs = net(inputs)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net
        '''

        output = net(image)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net
        output = output.detach().numpy()[0,:,:,:]

        # aplicarea regulii de decizie, pe baza iesirii modelului
        output = apply_regula_decizie_NCR_ET_ED(output)

        # salvare outputs
        output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output = cv2.flip(output, 0)
        cv2.imwrite(SAVE_PATH + '\\' + str(i) + '.png', output)

        no_examples = i

    #print('Incercence successful for the ' + str(i + 1) + ' examples')
    net.train()

"""
def inference_on_dataset_2doutput(net, inference_generator, SAVE_PATH):
    net.eval()
    for i, data in enumerate(inference_generator):
        image, label = data
        ########
        # CE TINE DE CUDA
        ########
        '''
        inputs = image.to(device).float()
        labels = label.to(device)
        # inputs, labels = data[0].to(device), data[1].to(device)

        # Se seteaza la ZERO gradientii optimkizarii (care erau nenuli de la iteratia precedenta)
        optimizer.zero_grad()
        # #print(inputs.shape)
        outputs = net(inputs)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net
        '''

        output = net(image)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net
        output = output.detach().numpy()[0,:,:,:]
        ##print(output)
        #print(np.min(np.min(output[0,:,:])),np.max(np.max(output[0,:,:])))
        #print(np.min(np.min(output[1,:,:])),np.max(np.max(output[1,:,:])))
        output[0,:,:] = output[0,:,:]/100000
        #print(np.min(np.min(output[0,:,:])),np.max(np.max(output[0,:,:])))
        output = np.argmax(output, axis=0)
        #print('k')
        #print(np.min(np.min(output[10:230,10:230])),np.max(np.max(output[10:230,10:230])))
        #print(np.min(np.min(output[10:230,10:230])),np.max(np.max(output[10:230,10:230])))
        #print('k')

        # se rescaleaza imaginile, pt ca sunt intre 0 si 1
        output = np.uint8(np.clip(output*255, a_min=0, a_max=255))

        # salvare outputs
        cv2.imwrite(SAVE_PATH + '\\' + str(i) + '.png', output)

        no_examples = i

    #print('Incercence successful for the ' + str(i + 1) + ' examples')
    net.train()
"""


def main():
    # pentru optimizatori
    import torch.optim as optim
    '''
    ########
    # CE TINE DE CUDA
    ########


    # eliberare cache
    torch.cuda.empty_cache()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    '''
    ########
    # PATHS
    ########

    INPUT_PATH = r'E:\an_4_LICENTA\Workspace\inference_workspace\input'
    SAVE_PATH = r'E:\an_4_LICENTA\Workspace\inference_workspace\output_semantic' 
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5_nomalizare0_mean_1_std.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\FULLDTS_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\FULLDTS_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61_0.878.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\ABSOLUTE_FULL_DTS_COR_unet_semantic_brats_2D_v3_COPIE__64_128_256_512_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.pth'
    # Best:
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi.pth'
    
    #
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\ABSOLUTE_FULL_DTS_COR_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.pth' # POSIBIL SA NU FIEE CINE CRERD EU CA E 

    ########
    # CREATING DATA GENERATORS
    ########

    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 1, # e important ca batrch_size sa fie 1!
              'shuffle': False,
              'num_workers': 0} # obligatoriu, num_workers=0. Altfel, paralelizarea proceselor face ca pozele sa nu fie in ordine. 0 inseamana ca doar procesul principal face task-ul.

    # se defineste dataset-ul pe care se vrea infetrenta
    #inference_set = Dataset(INPUT_PATH, normalize='0_mean_1std', zoom=False, rotate=False)
    inference_set = Dataset(INPUT_PATH, normalize=None, zoom=False, rotate=False, gauss_disp=0)

    # se instantiaza generatorul:
    inference_generator = torch.utils.data.DataLoader(inference_set, **params)

    ########
    # SELECT NETWORK
    ########
    net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[32, 64, 128, 256])
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=4, features=[64, 128, 256, 512])

    # se asigura precizia double a parametrilor
    net = net.double()

    ########
    # CE TINE DE CUDA
    ########
    # net.to(device)

    ########
    ## Incarcare model
    ########
    # checkpoint = torch.load(LOAD_PATH)
    # net, optimizer, start_epoch, loss, current_best_acc  = load_complete_model(checkpoint, LOAD_PATH, net, optimizer)
    optimizer = optim.Adam(net.parameters(), lr=0.01) # l-am pus doar ca sa mearga linia d emai jos de incarcare a modellui
    net, _, _, _, _ = load_complete_model(net, optimizer, LOAD_PATH)
    # net, optimizer, start_epoch, losslogger = load_checkpoint(net, optimizer, losslogger, filename = LOAD_PATH)

    inference_on_dataset_1doutout_semantic(net, inference_generator, SAVE_PATH)
    #inference_on_dataset_2doutput(net, inference_generator, SAVE_PATH)


if __name__ == '__main__':
    main()


































