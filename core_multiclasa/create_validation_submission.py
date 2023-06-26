
'''

2 pasi:

Pas1: generare predictii png-uri

Pas2: conversie png - .nii

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
from networks_folder_semantic.unet_semantic_brats_2D_v4_pls import UNET_semantic_brats_2D_v4
import tqdm
import cv2
import re

import nibabel as nib

def convert_png_to_nii(INPUT, OUTPUT):
    # toate slice-urile sunt inr angeul [0,155)
    
    ORIGINAL_NII_DIR = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\RSNA_ASNR_MICCAI_BraTS2021_ValidationData'
    # Voi extrage toate numerele de pacienti
    pattern = re.compile(r'seg_BraTS2021_([0-9]{5})_slice_\d+\.png')
    #print(pattern)

    x_numbers = []

    for filename in os.listdir(INPUT):
        match = pattern.match(filename)
        if match:
            x_numbers.append(int(match.group(1)))
    
    pacient_numbers = sorted(np.unique(x_numbers))
    pacient_numbers = [f'{num:05}' for num in pacient_numbers]
    #print(pacient_numbers)

    for pacient_number in pacient_numbers:
        # iterare prin pacienti

        # se incarca .nii-ul original pt t1, apoi se da overrite la fiecare slice in parte
        # fac asta pentru a avea orientarea initiala
        original_nii_path = ORIGINAL_NII_DIR + '\\' + 'BraTS2021_' + pacient_number + '\\' + 'BraTS2021_' + str(pacient_number) + '_t1.nii'
        nii_scan = nib.load(original_nii_path)
        data = nii_scan.get_fdata()
        #print(nii_scan) #img = scan_t1[:,:,i]

        for slice in range(155):
            filename = r'seg_BraTS2021_' + pacient_number + '_slice_' + str(slice) + '.png'
            #print('start--------------------------------------------------')
            #print(filename)
            #print('stop---------------------------------------------------')
            filepath = INPUT + '\\' + filename

            pred = cv2.imread(filepath)

            pred = pred[:,:,0]
            pred = np.where(pred[:, :] == 255, 4, pred[:, :]) # 4 este label-ul ET
            pred = np.where(pred[:, :] == 63, 1, pred[:, :])
            pred = np.where(pred[:, :] == 127, 2, pred[:, :]) # toate label-urile nenume reprezinta WT
            
            data[:,:,slice] = pred[:,:]

            

        #print(np.unique(data))

        # se reconstruieste imagine a, adaugand acelasi header si aceeai tr affina
        new_nii_scan = nib.Nifti1Image(data, nii_scan.affine, nii_scan.header)

        # Save the new NIfTI file
        nib.save(new_nii_scan, OUTPUT + '\\' + 'seg' + pacient_number)

            
            


def apply_regula_decizie_NCR_ET_ED(output):
    '''
        Defineste regula dupa care sunt interperetate iesirile retelei

        !! Folosita in cazul in care iesirele sunt, in aceasta ordine: NCR, ET, ED

        REGULA:
        voxcelul se atribuie clasei stratului de activare maxima, daca trece de pragul de 0.5
    '''
    # Creez hartile asociate OUT-ului:
    #print('jhdddddddddddddddddddddddddddddddddddd')
    #print(output.shape)
    predicted_NCR = output[0, 0,:,:]
    predicted_ET = output[0, 1,:,:]
    predicted_ED = output[0, 2,:,:]

    h, w = predicted_NCR.shape

    traduced_output = torch.from_numpy(np.uint8(np.zeros((h, w)))).cuda()

    ####
    # VARIANTA 1 - NU MERGE, DIN CAUZA "and"
    ####
    
    traduced_output = torch.where(torch.logical_and(torch.logical_and(predicted_NCR[:,:] > 0.5, predicted_NCR[:,:] > predicted_ET[:,:]), predicted_NCR[:,:] > predicted_ED[:,:]), 63, traduced_output[:,:])
    traduced_output = torch.where(torch.logical_and(torch.logical_and(predicted_ED[:,:] > 0.5, predicted_ED[:,:] > predicted_ET[:,:]), predicted_ED[:,:] > predicted_NCR[:,:]), 127, traduced_output[:,:])
    traduced_output = torch.where(torch.logical_and(torch.logical_and(predicted_ET[:,:] > 0.5, predicted_ET[:,:] > predicted_ED[:,:]), predicted_ET[:,:] > predicted_NCR[:,:]), 255, traduced_output[:,:])
    
    ####
    # VARIANTA 2 - ineficienta
    ####
    #print(np.unique(predicted_ED))
    '''
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
    '''
    
    return traduced_output



def inference_on_dataset_1doutout_semantic(net, inference_generator, SAVE_PATH, saveaspng, saveprobabilities):
    '''
    saveaspng = True => se va salav rezultatele, ca .png-uri, la SAVE_PATH
    saveprobabilities = True => se salveaza in directorul special probabilitatile prezise de retea
    '''
    net.eval()
    for i, data in enumerate(inference_generator):
        image, filepath = data
        #print('calea este sub forma:')
        #print(filepath)
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
        
        #image_tensor = torch.from_numpy(image)
        # transfer pe CUDA
        image = image.cuda()

        output = net(image)  # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net
        #output = output.detach().numpy()[0,:,:,:]

        # aplicarea regulii de decizie, pe baza iesirii modelului
        output_decided = apply_regula_decizie_NCR_ET_ED(output)
        output_decided = output_decided.cpu().numpy()
        
        if saveaspng:
            # salvare outputs
            #print(filepath[0].split('\\')[-1])
            output_decided = cv2.rotate(output_decided, cv2.ROTATE_90_COUNTERCLOCKWISE)
            output_decided = cv2.flip(output_decided, 0)
            #print(SAVE_PATH + '\\' + 'seg_' + filepath[0].split('\\')[-1] + '.png')
            cv2.imwrite(SAVE_PATH + '\\' + 'seg_' + filepath[0].split('\\')[-1] + '.png', output_decided)

            if saveprobabilities:

                PROBABILITIES_SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_suibmission\output_raw_probabilities'
                cv2.imwrite(PROBABILITIES_SAVE_PATH + '\\' + 'seg_' + filepath[0].split('\\')[-1] + '_0' + '.png', np.uint8(255*output[0,0,:,:].detach().cpu().numpy()))
                cv2.imwrite(PROBABILITIES_SAVE_PATH + '\\' + 'seg_' + filepath[0].split('\\')[-1] + '_1' +'.png', np.uint8(255*output[0,1,:,:].detach().cpu().numpy()))
                cv2.imwrite(PROBABILITIES_SAVE_PATH + '\\' + 'seg_' + filepath[0].split('\\')[-1] + '_2' +'.png', np.uint8(255*output[0,2,:,:].detach().cpu().numpy()))

        no_examples = i

    #print('Incercence successful for the ' + str(i + 1) + ' examples')
    net.train()










def main():
    # pentru optimizatori
    import torch.optim as optim

    ########
    # PATHS
    ########
    
    # cu gibbsremoved
    INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\gibbsartremoved_PNG_Val_Official_Dataset'
    # datele de train
    #INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'
    # fara gibbsremoved
    #INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\PNG_Val_Official_Dataset'
    SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_suibmission\output_off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi' 
    #SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_suibmission\output_TRAINDATA'
    #SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_suibmission\outtest'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5_nomalizare0_mean_1_std.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\COR_ABSOLUTE_FULL_DTS_unet_semantic_brats_2D_v4_pls_32_64_128_256.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\ABSOLUTE_FULL_DTS__unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph68.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\models_test_semantic\model0.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\train1_unet_semantic_brats_2D_v3_COPIE_softmax_4ch_fcost_dice_cainPaper_weights_batch.pth'
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi.pth'
    

    ################################################################
    # PAS 1 : generare png-uri predictii

    '''
    ########
    # CREATING DATA GENERATORS
    ########
    
    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 1, # e important ca batrch_size sa fie 1!
              'shuffle': False,
              'num_workers': 0} # obligatoriu, num_workers=0. Altfel, paralelizarea proceselor face ca pozele sa nu fie in ordine. 0 inseamana ca doar procesul principal face task-ul.

    # se defineste dataset-ul pe care se vrea infetrenta
    #inference_set = Dataset(INPUT_PATH, normalize='0_mean_1std', zoom=False, rotate=False)
    inference_set = UnlabeledDataset(INPUT_PATH, normalize=None, returnpath=True)

    # se instantiaza generatorul:
    inference_generator = torch.utils.data.DataLoader(inference_set, **params)

    ########
    # SELECT NETWORK
    ########
    net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[32, 64, 128, 256])
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=3, features=[64, 128, 256])
    #net = UNET_semantic_brats_2D_v4(in_channels=4, out_channels=3, features = [32, 64, 128, 256])

    # se asigura precizia double a parametrilor
    net = net.double()

    ########
    # CE TINE DE CUDA
    ########

    # eliberare cache
    torch.cuda.empty_cache()
    
    # CUDA for PyTorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    

    ########
    ## Incarcare model
    ########
    # checkpoint = torch.load(LOAD_PATH)
    # net, optimizer, start_epoch, loss, current_best_acc  = load_complete_model(checkpoint, LOAD_PATH, net, optimizer)
    optimizer = optim.Adam(net.parameters(), lr=0.01) # l-am pus doar ca sa mearga linia d emai jos de incarcare a modellui
    net, _, _, _, _ = load_complete_model(net, optimizer, LOAD_PATH)
    net.to(device)
    # net, optimizer, start_epoch, losslogger = load_checkpoint(net, optimizer, losslogger, filename = LOAD_PATH)

    inference_on_dataset_1doutout_semantic(net, inference_generator, SAVE_PATH, saveaspng=True, saveprobabilities=False)
    #inference_on_dataset_2doutput(net, inference_generator, SAVE_PATH)
    '''
    ################################################################
    # PAS 2 : Conversie png-uri, in .nii
    
    INPUT_PNG_PRED_DIR = r"E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_suibmission\output_off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi"
    OUTPUT_NII_PRED_DIR = r"E:\an_4_LICENTA\Workspace\Dataset\Val_Official_Dataset\zOUTPUT_for_submission_NII\output_off_ABSOLUTE_FULL_DTS_COR_transferLR_dezghetare_doar_ultimele_2_straturi"
    convert_png_to_nii(INPUT_PNG_PRED_DIR, OUTPUT_NII_PRED_DIR)
    



    

if __name__ == '__main__':
    main()













































