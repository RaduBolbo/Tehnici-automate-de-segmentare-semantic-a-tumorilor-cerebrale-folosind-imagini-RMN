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



def main():
    # pentru optimizatori
    import torch.optim as optim
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

    ########
    # PATHS
    ########
    
    #TEST_DATASET_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset'
    TEST_DATASET_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset'
    #TEST_DATASET_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Val_Dataset' # validare
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\FULLDTS_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61_0.878.pth'

    # 32:
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\ABSOLUTE_FULL_DTS_COR_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.pth'
    # 64:
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\ABSOLUTE_FULL_DTS_COR_unet_semantic_brats_2D_v3_COPIE__64_128_256_512_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.pth'
    # BEST:
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi.pth'

    ########
    # CREATING DATA GENERATORS
    ########

    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 1, # e important ca batrch_size sa fie 1!
              'shuffle': False,
              'num_workers': 0} # obligatoriu, num_workers=0. Altfel, paralelizarea proceselor face ca pozele sa nu fie in ordine. 0 inseamana ca doar procesul principal face task-ul.

    # se defineste dataset-ul pe care se vrea infetrenta
    #inference_set = Dataset(INPUT_PATH, normalize='0_mean_1std', zoom=False, rotate=False)
    test_set = Dataset(TEST_DATASET_PATH, normalize=None, zoom=False, rotate=False, gauss_disp=0)

    # se instantiaza generatorul:
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    ########
    # SELECT NETWORK
    ########
    # 32:
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[32, 64, 128, 256])
    # 64:
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[64, 128, 256, 512])
    net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=4, features = [64, 128, 256, 512])
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[64, 128, 256])

    # se asigura precizia double a parametrilor
    #net = net.double()
    # se trece in modul de evaluare
    net.eval()

    ########
    # CE TINE DE CUDA
    ########
    net.to(device)

    ########
    ## Incarcare model
    ########
    # checkpoint = torch.load(LOAD_PATH)
    # net, optimizer, start_epoch, loss, current_best_acc  = load_complete_model(checkpoint, LOAD_PATH, net, optimizer)
    optimizer = optim.Adam(net.parameters(), lr=0.0001) # l-am pus doar ca sa mearga linia d emai jos de incarcare a modellui
    net, _, _, _, _ = load_complete_model(net, optimizer, LOAD_PATH)
    # net, optimizer, start_epoch, losslogger = load_checkpoint(net, optimizer, losslogger, filename = LOAD_PATH)
    net.eval()

    # testarea
    current_dice_score_ET, current_dice_score_NCR, current_dice_score_ED, current_decided_dice_score_WT, current_decided_dice_score_TC, current_decided_dice_score_ET, senz_WT, senz_TC, senz_ET, spec_WT, spec_TC, spec_ET, hausdorff95_WT, hausdorff95_TC, hausdorff95_ET = test_network(net, device, test_generator, verboise=False, request_Hausdorff_senz_spec=True)

     







if __name__ == '__main__':
    main()




