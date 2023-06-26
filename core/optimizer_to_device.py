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



def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim