'''
* Este o versiune cu transfer LR a primei retele

UNET_binar_brats_2D_v1, INCLUSIV CU MECANISMUL DE ATERNTII

STRATURI DE OUTPUT:
    0 = WT
    1 = ET
    2 = NCR

'''

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare
import torch
import numpy as np
import torch.optim as optim #ttrebuie daor pt a-l tr ca arg
# pentru incarcarea retelelor:
#from ..load_checkpoint import load_complete_model

# reteaua originala
#from unet_binar_brats_2D_v3 import UNET_binar_brats_2D_v3
# ar treb sa mearga si cu asata
#from unet_semantic_brats_2D_v3_COPIE import UNET_semantic_brats_2D_v3_COPIE
#from unet_semantic_brats_2D_v3_COPIE import UNET_semantic_brats_2D_v3_COPIE
from load_checkpoint import load_complete_model


def get_modified_network(net, mode, verboise):
    '''
    mode = 'last_layer' => se dezgheata doar ultimul strat
    mode = 'entire_decoder' => se dezgheata tot decoderul
    '''
    MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_norarsimpla_bil0.9359.pth'

    #MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_simplu_32_64_128_256_nepreprocesat_lr10^-4_nomalizaresimpla.pth'
    #MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_norarsimpla_bil0.9359.pth'
    #MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_norarsimpla_bil0.9359.pth'
    #MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_norarsimpla_bil0.9359.pth'


    #MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_semantica\FULLDTS_unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61_0.878.pth'

    # reteaua originala
    #net = UNET_binar_brats_2D_v3(in_channels=4, out_channels=1, features = [32, 64, 128, 256])
    # ar treb sa mearga si cu asata
    #net = UNET_semantic_brats_2D_v3_COPIE(in_channels=4, out_channels=1, features = [32, 64, 128, 256])
    optimizer = optim.Adam(net.parameters(), lr=0.0001) # DAR NU CONTEAZA!
    pretrained_net, _, _, _, _ = load_complete_model(net, optimizer, MODEL_PATH)

    

    if verboise:
        print('')
        print('straturile ce necesita gradient')
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            if param.requires_grad:
            #   print(name, param.data)
                print(name)
                c += 1
        print(c)

        
        print('')
        print('toarte straturile')
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            c += 1
        print(c)


    # adaptorul la numarul de clase
    adapter_layer = nn.Conv2d(32*3, 3, kernel_size=1, padding = 'same')
    # voi inlocui ultimul strat al retelei, cu unul nou
    pretrained_net.adapter_layer = adapter_layer


    '''
    c = 0
    for name, param in pretrained_net.named_parameters():
        print(name)
        c += 1
    print(c)
    print('')
    '''

    # inghetarea tuturorstraturilor
    for name, param in pretrained_net.named_parameters():
        param.requires_grad = False

    if mode == 'entire_decoder':
        # dezghetarea unor straturi
        for name, param in pretrained_net.named_parameters():
            if name[0:13] == 'adapter_layer':
                param.requires_grad = True
            if name[0:8] == 'ups_conv':
                param.requires_grad = True
            if name[0:10] == 'ups_expand':
                param.requires_grad = True
            if name[0:22] == 'squeeze_and_extraction':
                param.requires_grad = True
            if name[0:10] == 'bottleneck':
                param.requires_grad = True
    elif mode == 'last_layer':
        for name, param in pretrained_net.named_parameters():
            if name[0:13] == 'adapter_layer':
                param.requires_grad = True


    for name, param in pretrained_net.named_parameters():
        print(name, param.requires_grad)


    if verboise:
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            c += 1
        print(c)

    return pretrained_net

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

def get_custom_modified_network(net, mode, verboise):
    '''
    mode = 'last_layer' => se dezgheata doar ultimul strat
    mode = 'entire_decoder' => se dezgheata tot decoderul
    mode = 'last_2_layers'
    '''
    MODEL_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_norarsimpla_bil0.9359.pth'

    optimizer = optim.Adam(net.parameters(), lr=0.0001) # DAR NU CONTEAZA!
    pretrained_net, _, _, _, _ = load_complete_model(net, optimizer, MODEL_PATH)

    

    if verboise:
        print('')
        print('straturile ce necesita gradient')
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            if param.requires_grad:
            #   print(name, param.data)
                print(name)
                c += 1
        print(c)

        
        print('')
        print('toarte straturile')
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            c += 1
        print(c)


    # adaptorul la numarul de clase
    adapter_layer = nn.Conv2d(32*3, 3, kernel_size=1, padding = 'same')
    # voi inlocui ultimul strat al retelei, cu unul nou
    pretrained_net.adapter_layer = adapter_layer


    '''
    c = 0
    for name, param in pretrained_net.named_parameters():
        print(name)
        c += 1
    print(c)
    print('')
    '''

    # inghetarea tuturorstraturilor
    for name, param in pretrained_net.named_parameters():
        param.requires_grad = False

    if mode == 'entire_decoder':
        # dezghetarea unor straturi
        for name, param in pretrained_net.named_parameters():
            if name[0:13] == 'adapter_layer':
                param.requires_grad = True
            if name[0:8] == 'ups_conv':
                param.requires_grad = True
            if name[0:10] == 'ups_expand':
                param.requires_grad = True
            if name[0:22] == 'squeeze_and_extraction':
                param.requires_grad = True
            if name[0:10] == 'bottleneck':
                param.requires_grad = True
    elif mode == 'last_layer':
        for name, param in pretrained_net.named_parameters():
            if name[0:13] == 'adapter_layer':
                param.requires_grad = True
    elif mode == 'last_2_layers':
        for name, param in pretrained_net.named_parameters():
            if name[0:13] == 'adapter_layer':
                param.requires_grad = True
            if name[0:10] == 'ups_conv.3':
                print(name)
                param.requires_grad = True
            if name[0:24] == 'squeeze_and_extraction.3':
                param.requires_grad = True


    if verboise:
        c = 0
        for name, param in pretrained_net.named_parameters():
            print(name)
            c += 1
        print(c)

    return pretrained_net

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################


def unfreeze_whole_network(net):
    # dezghetarea tuturorstraturilor
    for name, param in net.named_parameters():
        param.requires_grad = True
    print('-> All layers unfreezed')
    return net



def test():
    get_modified_network(False)

if __name__ == "__main__":
    test()






