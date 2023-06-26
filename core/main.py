
########
# INPORTS
########

from tabnanny import verbose
from dataloader import *
#from retele import *
from train1 import *

import torch
torch.manual_seed(0)
import torchvision # contine data loader-uri pentru seturi de date comune
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np


# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru incarcarea retelelor:
from load_checkpoint import load_complete_model

# pentru punerea optimizatorului pe GPU:
from optimizer_to_device import optimizer_to

###### import-uri pentru retele
#from networks import *
from networks_folder.unet_model import UNET
from networks_folder.unet_binar_brats_2D_v1 import UNET_binar_brats_2D_v1
from networks_folder.unet_binar_brats_2D_v2 import UNET_binar_brats_2D_v2
from networks_folder.unet_binar_brats_2D_v3 import UNET_binar_brats_2D_v3
from networks_folder.unet_binar_brats_2D_v4 import UNET_binar_brats_2D_v4
from networks_folder.unet_binar_brats_2D_v5_gauss import UNET_binar_brats_2D_v5_gauss
from networks_folder.unet_binar_brats_2D_v6_plus import UNET_binar_brats_2D_v6_plus

# pentru studierea poolingului
from networks_folder.unet_binar_brats_2D_v6_AvgPoolBlock import UNET_semantic_brats_2D_v6


from networks_folder.segnet_semantic_brats_2D_v1 import Segnet_semantic_brats_v1

from networks_folder.unet_resnet_hybrid_1 import UNET_RESNET_HYBRID_1
from networks_folder.unet_resnet_hybrid_2 import UNET_RESNET_HYBRID_2

from networks_folder.unet_binar_brats_2D_v7_oricesizelafiltre_plus import UNET_binar_brats_2D_v7_plus

#from networks_folder.resnet_1 import ResNet1
from networks_folder.resnet_1_corectat import ResNet1
from networks_folder.resnet_2_nolongskip import ResNet2
from networks_folder.resnet_noskip import Plain_net

def main():

    # pentru optimizatori
    import torch.optim as optim

    # detectia de anomalii in graful computational
    torch.autograd.set_detect_anomaly(True)
    
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

    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\PNG_Validation_Dataset'
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Ultrareduced_Dataset\PNG_Validation_Dataset'
    SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\models_test' # * de aici schimb path-ul
    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\modele_salvate_binara\model28.pth'

    #### Pentru dataset simplu full
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Training_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Test_Dataset' # * de aici schimb path-ul

    #### Pentru dataset GibbsArtRemoverd full
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Val_Dataset'
    TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Test_Dataset' # * de aici schimb path-ul

    ######################################### DATADET REDUS #########################################

    #### Pentru dataset simplu REDUCED 1/3 NEPREPROCESAT 8b
    TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'
    VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Test_Dataset' # * de aici schimb path-ul

    #### Pentru dataset simplu REDUCED 1/3 NEPREPROCESAT 16b
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_16bnepreprocesat\16b_PNG_Test_Dataset' # * de aici schimb path-ul

    #### Pentru dataset simplu REDUCED 1/3 BIASSFIELDREMOVED 8b
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Val_Dataset'
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_PNG_Test_Dataset' # * de aici schimb path-ul
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Val_Dataset'

    #### Gibbs ringing asrtifact removal
    #TEST_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Test_Dataset' # * de aici schimb path-ul
    #TRAIN_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Train_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_gibbsringingartefactremoved\gibbsringingartefactremoval_PNG_Val_Dataset'
    #VAL_DATA_PATH = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Val_Dataset'

    

    ########
    # CREATING DATA GENERATORS
    ########



    # se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
    params = {'batch_size': 6,
              'shuffle': True,
              'num_workers': 1}
    #nr_epochs = 50 # * de aici se schimba nr de epoci
    nr_epochs = 50

    # s einstantiaza daatseturile
    #training_set = Dataset(TRAIN_DATA_PATH, normalize='0_mean_1std', zoom = False, rotate = False)
    #val_set = Dataset(VAL_DATA_PATH, normalize='0_mean_1std', zoom = False, rotate = False)

    #training_set = Dataset(TRAIN_DATA_PATH, normalize=None, zoom = True, rotate = False, horiz_flip=True, vert_flip=True, gauss_disp=0.01) # best variant
    training_set = Dataset(TRAIN_DATA_PATH, normalize=None, zoom = False, rotate = False, horiz_flip=False, vert_flip=False, gauss_disp=0) # simplu
    val_set = Dataset(VAL_DATA_PATH, normalize=None, zoom = False, rotate = False, horiz_flip=False, vert_flip=False, gauss_disp=0)

    # se instantiaza generatorii:
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    val_generator = torch.utils.data.DataLoader(val_set, **params)

    ########
    # SELECT NETWORK
    ########
    #net = Net_test()
    #net = UNET(in_channels=4, out_channels=1, features = [64, 128, 256, ])
    #net = UNET(in_channels=4, out_channels=1, features = [8, 16, 32, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [8, 16, 32, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=2, features = [8, 16, 32, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [64, 128, 256, ])
    #net = UNET_binar_brats_2D_with_attenstions()

    # Variere dimensiuni. Se vor incerca:
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [32, 64, ])
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [16, 32, 64, ]) # MAX: 0.910
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [32, 64, 128, ]) # MAX: 0.914
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [64, 128, 256, ]) # MAX: 0.65
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [celmaibun, celmaibun, celmaibun, celmaibun*2])
    # se va testa ceva intre cele doua cele mai bune; cea cu 32 si cea cu 16:
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [25, 50, 100, ]) # prima epoca da un loss mai prost, dar un val_dice mai bun. Poate merita incercat
    # as putea incerca sa testez daca adancirea retrelkei care incepe cu 16 e mai buna ca cea a retelei care incepe cu 32, ca sa vad daca retelele mai adanci generalizeaza mai bine sdi memoreaza mai prost
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [16, 32, 64, 128]) # MAX: 0.9166
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # MAX ? pe lr=10^(-3); MAX 0.927 pe lr=10^(-4); MAX 0.920 pe lr=10^(-5)
    #net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [32, 64, 128, 256, 512])
    

    # Varianta cu atentiile din paper (Squeeze & Extract)
    #net = UNET_binar_brats_2D_v3(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # MAX: 0.929 pe lr=10^(-5)sau -4???
    #net = UNET_binar_brats_2D_v3(in_channels=4, out_channels=1, features = [32, 32, 32, 32])

    # Investigare dimesniune nucleu Average Pool 2D
    #net = UNET_semantic_brats_2D_v6(in_channels=4, out_channels=1, features = [32, 64, 128, 256])

    # varianta the best
    #net = UNET_binar_brats_2D_v3(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # MAX: 0.929 pe lr=10^(-5)sau -4??? 
    # varianta the best+ gasussian
    #net = UNET_binar_brats_2D_v5_gauss(in_channels=4, out_channels=1, features = [32, 64, 128, 256])
    
    #net = UNET_binar_brats_2D_v3(in_channels=4, out_channels=1, features = [40, 80, 160, 320])

    # Varianta cu plus
    #net = UNET_binar_brats_2D_v6_plus(in_channels=4, out_channels=1, features = [32, 64, 128, 256])

    # Oricesize
    #net = UNET_binar_brats_2D_v7_plus(in_channels=4, out_channels=1, features = [32, 32, 32, 32])

    # Varianta de UNet ResNet hgibrid
    #net = UNET_RESNET_HYBRID_1(in_channels=4, out_channels=1, features = [32, 64, 128, 256], number_of_blocks=2) # => 0.902
    #net = UNET_RESNET_HYBRID_1(in_channels=4, out_channels=1, features = [32, 64, 128, 256], number_of_blocks=3) # => ?
    #net = UNET_RESNET_HYBRID_2(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # => ?


    # Variere adancime blocuri

    # se asigura precizia double a parametrilor
    # net = UNET_binar_brats_2D_v2(in_channels=4, out_channels=1, features = [32, 64, 128, 256]) # diferenta infima

    # SegNet
    # Varianta SegNet
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [32, 64, 128, 256])
    net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [64, 128])
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [64, 128, 256]) # best sofar # de ce la binar nu mai merge bine?
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [64, 128, 256, 512]) # dezastrtuos. De ce pe binar megrge????
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [64, 128, 256, 512, 1024]) 
    #net = Segnet_semantic_brats_v1(in_channels=4, out_channels=1, features = [128, 256, 512])
    
    #net = net.double()
    #net = net.float()

    # ResNet
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32]) # 0.862 cu loss 0.105
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32]) # 0.878 cu loss 0.091 / corectat => 0.857 cu loss 0.113
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32]) # 0.887 cu loss 0.084 / corectat =>
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # ? cu loss ?
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # / corectat => 0.874 cu loss 0.090
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # / corectat => 0.886 cu loss 0.091; 12:30 min
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # 32fx24 / corectat => 0.896 cu loss 0.085
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # 32fx48
    #net = ResNet1(in_channels=4, out_channels=1, features = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]) # / corectat => 0.905 cu loss ?
    #net = ResNet1(in_channels=4, out_channels=1, features = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
    #net = ResNet1(in_channels=4, out_channels=1, features = [32, 32, 64, 64, 128, 128]) # ? cu loss ?

    #net = ResNet2(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # noskip 32fx24 / corectat =>  cu loss 
    #net = Plain_net(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]) # plain 32fx24 / corectat =>  cu loss 

    #net = Plain_net(in_channels=4, out_channels=1, features = [32, 32, 32, 32, 32, 32])


    ########
    # SELECT OPTIMIZER
    ########
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001) # practic nu functioneaza
    #optimizer = optim.Adam(net.parameters(), lr=0.0001) # e mai bine, cel putin primele 40 de epoci
    optimizer = optim.Adam(net.parameters(), lr=0.00001) # cu 10^(-5) se antrena ok pe UNET_simplu
    #optimizer = optim.Adam(net.parameters(), lr=0.000001) # e mai rau

    ########
    # SELECT SCHEDULER
    ########
    # 1) ExponentialLR:
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # scheduler-ul modifica LR-ul, in timpul antrenarii

    # 2) StepLR: la fiecare step_size epoci, lr-ul se inmulteste (decay) cu gamma.
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, verbose = True)
    # IDEE DE BAZA: AR TREBUI SA SCAD LR-UL DOAR ATUNCI CAND LOSS-UL NU MAI SCADE. LA EPOCA IN CARE LOSS-UL SE PLAFONEAZA, SE SCADE LR-ul. NU ARE SENS LA 20 DE EPOCI - ATAT DE DES
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose = True)



    ########
    # CE TINE DE CUDA
    ########
    current_best_score = 0 # in cazul in care inca nu s-a intrat in functia load_complete_model(), acuratetea initiala e 0%
    # ma asigur ca se incepe de la epoca 0. Va fi suprascris daca se incarca modelul
    start_epoch = 0
    #net.to(device)

    ########
    ## Incarcare model
    ########
    #net, optimizer, start_epoch, loss, current_best_score = load_complete_model(net, optimizer, LOAD_PATH)
    net.to(device)
    # optimizatorul este pus pe CPU dupa ce e incarcat de functia load_complete_model
    optimizer = optimizer_to(optimizer, device)


    #net = train_network(net, device, training_generator, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_acc, mod = 'complete_save')
    net = train_network(net, device, training_generator, val_generator, nr_epochs, SAVE_PATH, optimizer, scheduler, current_best_score, start_epoch, mod = 'complete_save') # MAX:0.93






if __name__ == '__main__':
    main()







