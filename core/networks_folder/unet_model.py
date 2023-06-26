
'''
PILOT DE UNET

este un Unet 2D simplu, fara mecanisme de atentie, pentru clasificare binara
Spre deosebire de UNET-ul descris in "Brats_with_attentions_2D", este:
1. Fara mecanismul de 'atentii'. In locul sa, se implementeaza sipla concatenare
2. este pentru clasificare bianra
3. Sunt unele diferente la blocul portocaliu de upsampeling. Practic, in locul sau este ConvTranspose2d, cu kernel de 2 si stride de 2
4. Nu are acel 'residual unit' peste orice bloc de 2 convolutii
5. Nu are blocul verde ce conv2d cu 2x2 cu stride de 2. In locul sau, are maxpooling

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
            nn.ReLU(inplace=True), #! DE AI CI AR PUTEA VENI O EROARE. Il pun pentru a elibera o parte din memorie. Asta inseamna ca nu se va rea un spatiu aditional pentru rezultatul BatchNorm, ci se va trece direct prin ReLU
            nn.Conv2d(out_channels, out_channels, 3, 1, bias=False, padding ='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features = [64, 128, 256, ]):
        '''
        :param in_channels: number of input chanels in the UNET
        :param out_channels: number of classes OR 1, in case of a binary classification
        :param features: the numer of channels specific to each convolutional block
        '''
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # se va crea UNET-ul:

        # Up part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # inputul etse in_channels, iar la output e cat specifica feature
            in_channels = feature # ma asigur ca la iteratia viitoare, voi avea practiv un fel de features[i-1]

        # Down part
        for feature in reversed(features):
            # partea de upsampling -> o fac, in cazul asta, prind dublarea dimensionalitatii cu o ConvTranspinse2d
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2) # se va DUBLA inaltimea si latimea imaginii, prin kernel 2 si stride 2
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # spatiul latent
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # adaptorul la numarul de clase
        self.adapter_layer = nn.Conv2d(features[0], out_channels, kernel_size=1, padding = 'same')

    def forward(self, x):
        skip_connections = []

        # 1. se trece prin partea de down
        # 2. se inregistreaz acare au fost toutput-urile blocurilor convolutionale in partea de down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # se trece prin spatiul latent
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # inverseaza ordinea vectorului
        for i in range(0, len(self.ups), 2): # se merge cu PASUL DE 2 deoarece syunt 2 convolutii
            x = self.ups[i](x) # aici este upsample-ul
            skip_connection = skip_connections[i//2] # totusi, skip_connection-urile nu se parcurg cu aps de 2s

            # PROTECTIE
            # ? nu e o problema?Nu sparge graful computational?
            '''
            if x.shape != skip_connection.shape:
                #print(x.shape)
                #print(skip_connection.shape[2:])
                x.reshape(skip_connection.shape[2:])
                #x.reshape(skip_connection.shape)
                #x.view(skip_connection.shape)
                #x = F.resize(x, size = skip_connection.shape[2:])
            '''
            concatenated_x = torch.cat((skip_connection, x), dim = 1) # concatenarea x cu skip, de-a lungul axei canalelor (dim=1)
            x = self.ups[i+1](concatenated_x)

        x = self.adapter_layer(x)
        x = F.sigmoid(x)

        return x

def test():
    x = torch.randn((4,4,240,240))
    model = UNET(4, 1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)

if __name__ == "__main__":
    test()








