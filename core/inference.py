
'''

'''


########
# INPORTS
########

from tabnanny import verbose
from dataloader import *
# from retele import *
from train1 import *

import torch
import numpy as np

torch.manual_seed(0)

# pentru incarcarea retelelor:
from load_checkpoint import load_complete_model

###### import-uri pentru retele
# from networks import *
from networks_folder.unet_model import UNET
from networks_folder.unet_binar_brats_2D_v1 import UNET_binar_brats_2D_v1
import tqdm
import cv2


def inference_on_dataset_1doutout(net, inference_generator, SAVE_PATH):
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
        output = output.detach().numpy()[0,0,:,:]

        # se rescaleaza imaginile, pt ca sunt intre 0 si 1
        output = np.uint8(np.clip(output*255, a_min=0, a_max=255))

        # salvare outputs
        cv2.imwrite(SAVE_PATH + '\\' + str(i) + '.png', output)

        no_examples = i

    #print('Incercence successful for the ' + str(i + 1) + ' examples')
    net.train()

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
    SAVE_PATH = r'E:\an_4_LICENTA\Workspace\inference_workspace\output' 
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5.pth'
    #LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_64_128_255_nepreprocesat_lr10^-5_nomalizare0_mean_1_std.pth'
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate\UNET_simplu_32_64_128_256_nepreprocesat_lr10^-5_nomalizaresimpla.pth'

    ########
    # CREATING DATA GENERATORS
    ########

    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}

    # se defineste dataset-ul pe care se vrea infetrenta
    #inference_set = Dataset(INPUT_PATH, normalize='0_mean_1std', zoom=False, rotate=False)
    inference_set = Dataset(INPUT_PATH, normalize=None, zoom=False, rotate=False)

    # se instantiaza generatorul:
    inference_generator = torch.utils.data.DataLoader(inference_set, **params)

    ########
    # SELECT NETWORK
    ########
    # net = Net_test()
    # net = UNET(in_channels=4, out_channels=1, features = [64, 128, 256, ])
    #net = UNET(in_channels=4, out_channels=1, features=[8, 16, 32, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features=[8, 16, 32, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=2, features=[8, 16, 32, ])
    # net = UNET_binar_brats_2D_with_attenstions()

    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features=[64, 128, 256, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features=[32, 64, 128, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features=[16, 32, 64, 128, ])
    net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features=[32, 64, 128, 256, ])

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

    inference_on_dataset_1doutout(net, inference_generator, SAVE_PATH)
    #inference_on_dataset_2doutput(net, inference_generator, SAVE_PATH)


if __name__ == '__main__':
    main()


































