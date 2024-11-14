from ann_visualizer.visualize import ann_viz;

# pentru incarcarea retelelor:
from core.load_checkpoint import load_complete_model

from core.networks_folder.unet_binar_brats_2D_v1 import UNET_binar_brats_2D_v1

import torch.optim as optim

def main():

    LOAD_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\models_test\model0.pth'

    net = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [8, 16, 32, ])
    #net = UNET_binar_brats_2D_with_attenstions()

    # se asigura precizia double a parametrilor
    net = net.double()

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    ########
    # SELECT SCHEDULER
    ########
    # 1) ExponentialLR:
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # scheduler-ul modifica LR-ul, in timpul antrenarii

    # 2) StepLR: la fiecare step_size epoci, lr-ul se inmulteste (decay) cu gamma.
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7, verbose = True)



    current_best_score = 0 # in cazul in care inca nu s-a intrat in functia load_complete_model(), acuratetea initiala e 0%
    start_epoch = 0

    ########
    ## Incarcare model
    ########
    net, optimizer, start_epoch, loss, current_best_score = load_complete_model(net, optimizer, LOAD_PATH)


    ann_viz(net, title="My first neural network")

if __name__ == '__main__':
    main()

