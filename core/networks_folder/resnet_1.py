
'''
PILOT DE ResNet



'''

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare
import torch
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential( # e un CONTAINER SECVENTIAL, ce va contine straturi
            nn.Conv2d(in_channels, out_channels, 3, 1, bias=False, padding = 'same'), # bias-ul asociat va fi oricum eliminat de batch norm, deci e nenecesar
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True), #! DE AI CI AR PUTEA VENI O EROARE. Il pun pentru a elibera o parte din memorie. Asta inseamna ca nu se va rea un spatiu aditional pentru rezultatul BatchNorm, ci se va trece direct prin ReLU
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, bias=False, padding ='same'),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)

class Singlecov_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Singlecov_layer, self).__init__()
        self.conv = nn.Sequential( # e un CONTAINER SECVENTIAL, ce va contine straturi
            nn.Conv2d(in_channels, out_channels, 3, 1, bias=False, padding = 'same'), 
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)

class ResNet1(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features = [32, 32, 32, 32, ]):
        '''
        :param in_channels: number of input chanels in the UNET
        :param out_channels: number of classes OR 1, in case of a binary classification
        :param features: the numer of channels specific to each convolutional block
        '''

        super(ResNet1, self).__init__()

        self.input_layer = Singlecov_layer(in_channels, features[0])
        in_channels = features[0]

        self.straturi = nn.ModuleList()

        # se va crea ResNet-ul:
        for feature in features:
            self.straturi.append(DoubleConv(in_channels, feature)) # inputul etse in_channels, iar la output e cat specifica feature
            in_channels = feature # ma asigur ca la iteratia viitoare, voi avea practiv un fel de features[i-1]

        # stratul penultim de out
        self.penoutput_layer = Singlecov_layer(features[-1], features[-1])
        # adaptorul la numarul de clase
        self.adapter_layer = nn.Conv2d(features[-1], out_channels, kernel_size=1, padding = 'same')

    def forward(self, x):
        
        skip_connections = []

        x = self.input_layer(x)

        long_skip_connection = x

        for strat in self.straturi:
            skip_connections = x
            out = strat(x) # IMPORTANT! acum x va referi UN ALT TENSOR DIN MEMORIE CARE SE CREEAZA
            x = out + skip_connections

        # adaug trasaturile de la final
        out = x
        x = out+long_skip_connection
        x = self.penoutput_layer(x)

        #print(x.shape)
        x = self.adapter_layer(x)
        #print(x.shape)
        x = F.sigmoid(x)
        #print(x.unique)

        return x

def test():
    x = torch.randn((4,4,240,240))
    model = ResNet1(4, 1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)

if __name__ == "__main__":
    test()








