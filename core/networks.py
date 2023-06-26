import torch

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare


class Net_test(nn.Module):

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        # ? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.dropout = nn.Dropout(0.1)  # oare mai merge crescut?
        # self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(4, 16, 3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.convTranspose1 = nn.ConvTranspose2d(16, 1, 3)

        self.convcentru = nn.Conv2d(16, 16, 3, padding = 'same') # padding = same, ca sa ramana aceeasi dimensiune

        self.unpool = nn.MaxUnpool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(1) # rol in a normaliza outputul

        #self.output_activation_function = F.softmax()

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        # PARTEA I
        ##print('x shape: ' + str(x.shape))
        x, indices = self.pool(F.relu(self.conv1(x)))
        ##print('x shape: ' + str(x.shape))
        x = self.batchnorm1(x)

        # CENTRU
        ##print('x shape: ' + str(x.shape))
        x = self.convcentru(F.relu(x))

        # PARTEA II
        ##print('x shape: ' + str(x.shape))
        x = self.unpool(x, indices)
        ##print('x shape: ' + str(x.shape))
        x = F.relu(self.convTranspose1(x))
        ##print('x shape: ' + str(x.shape))

        #x = self.output_activation_function(x)
        #x = F.softmax(x)
        x = F.sigmoid(x)

        return x

###################################################################################################
######################################### UNET-uri ################################################
###################################################################################################


class Unet2D(nn.Module):

    '''
    E un model de baza de la care voi porni pentru a crea alte modele
    '''

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()



    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):

        return x


class Unet2D_simplu_1(nn.Module):

    '''
    este un Unet 2D simplu, fara mecanisme de atentie, pentru clasificare binara
    Este U-Net-ul dresris in "Brats_with_attentions_2D", DAR:
    1. Fara mecanismul de 'atentii'
    2. este pentru clasificare bianra
    3. Sunt unele diferente la blocul portocaliu de upsampeling, care nush exact ce vrea sa fie in papaer
    '''

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()


        #self.output_activation_function = F.softmax()

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):


        return x











