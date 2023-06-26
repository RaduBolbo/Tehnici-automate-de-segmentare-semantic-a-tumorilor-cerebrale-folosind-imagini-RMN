'''

UNET_binar_brats_2D_v1, INCLUSIV CU MECANISMUL DE ATERNTII
IN P{LUS, acest model are si un layer gaussian la intrare.}

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

class Shrinker(nn.Module):
    def __init__(self, feature):
        super(Shrinker, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature*2, feature*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(feature*2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, feature):
        super(Upsample, self).__init__()
        self.Upsample = nn.Sequential(
            # 1. Varianta de ConvTranspose2d
            #nn.ConvTranspose2d(6*feature, feature, kernel_size=2, stride=2)

            # 2. Varianta cu interpolare: se va testa care e cea mi buna
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(6*feature, feature, kernel_size=3, stride=1, padding = 'same')
            )
    def forward(self, x):
        return self.Upsample(x)
        
class Upsample_latent(nn.Module):
    def __init__(self, feature):
        super(Upsample_latent, self).__init__()
        self.Upsample = nn.Sequential(
            # 1. Varianta de ConvTranspose2d
            #nn.ConvTranspose2d(4*feature, feature, kernel_size=2, stride=2)
            
            # 2. Varianta cu interpolare: se va testa care e cea mi buna
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(4*feature, feature, kernel_size=3, stride=1, padding = 'same')
            )
    def forward(self, x):
        return self.Upsample(x)


class Input_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Input_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding = 'same'),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

class Squeeze_and_extraction_block(nn.Module):
    def __init__(self, feature):
        super(Squeeze_and_extraction_block, self).__init__()
        self.compute_channel_weights = nn.Sequential(
            # Global Average Pooling Layer
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Primul strat FC
            nn.Linear(2*feature, 2*feature),
            nn.BatchNorm1d(2*feature),
            nn.ReLU(inplace=True),

            # Al doilea si ultimul strat FC
            nn.Linear(2*feature, 2*feature),
            # Functia de activare sigmoida, pentru a obtine ponderi pty fiecare canal
        )

    def forward(self, x):
        return self.compute_channel_weights(x)

class UNET_binar_brats_2D_v5_gauss(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features = [64, 128, 256, ]):
        '''
        :param in_channels: number of input chanels in the UNET
        :param out_channels: number of classes OR 1, in case of a binary classification
        :param features: the numer of channels specific to each convolutional block
        '''
        super(UNET_binar_brats_2D_v5_gauss, self).__init__()
        self.ups_expand = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.squeeze_and_extraction = nn.ModuleList()
        #self.downs_conv = nn.ModuleList()
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.skrinker_conv = nn.ModuleList() # in loc de maxpooling

        # Layerul de zgomot gaussian
        self.input_gauss_layer = GaussianNoise(0.005)

        # pentru stratul d einput:
        self.input_layer = Input_layer(in_channels, features[0])
        #self.input_layer = Input_layer(in_channels, 8)

        # se va crea UNET-ul:

        # Down part
        in_channels_for_downs = features[0] # asta deoarece
        for feature in features:
            self.downs.append(DoubleConv(feature, feature)) # inputul etse in_channels, iar la output e cat specifica feature
            self.skrinker_conv.append(Shrinker(feature)) # se merge pe ideea ca merg din 2 in 2

        # Up part
        for feature in reversed(features):
            # partea de upsampling -> o fac, in cazul asta, prind dublarea dimensionalitatii cu o ConvTranspinse2d
            self.ups_expand.append(
                # 1. Varianta cu ConvTrasnpose2d
                #nn.ConvTranspose2d(feature*4, feature, kernel_size=2, stride=2) # se va DUBLA inaltimea si latimea imaginii, prin kernel 2 si stride 2
                # 2. Varianta cu interpolare: se va testa care e cea mi buna
                #trebuie facut un modul secvential cu cele doua
                Upsample(feature)
                #nn.Upsample(scale_factor = 2, mode = 'nearest'),
                # oare Conv2d ar trebui sa aiba kernel de 2 sau de 3? 
                #nn.Conv2d(feature, feature//4, kernel_size=3, stride=1, padding = 'same')
            )
            #
           # self.ups.append(DoubleConv(feature*2, feature))

        for feature in reversed(features):
            self.squeeze_and_extraction.append(
                Squeeze_and_extraction_block(feature)
            )

        for feature in reversed(features):
            self.ups_conv.append(DoubleConv(2*feature, feature))

        # spatiul latent
        self.bottleneck = DoubleConv(features[-1]*2, features[-1]*2)

        ############## ADAUGARE ################
        # upsample-ul din spatiul latent
        self.ups_expand_latent = Upsample_latent(features[-1])
        ############## ADAUGARE ################

        # adaptorul la numarul de clase
        self.adapter_layer = nn.Conv2d(features[0]*3, out_channels, kernel_size=1, padding = 'same')

        #self.testare = nn.Conv2d(4, 8, kernel_size=2, stride=1, padding = 'same')

    def forward(self, x):
        skip_connections = []

        # se trece input-ul prin stratul de zgomot gaussian
        x = self.input_gauss_layer(x)

        # Prima data se trece prin layer-ul de input, care sa mor daca stiu de ce e 2x2:
        x = self.input_layer(x)
        #x = self.testare(x)

        # 1. se trece prin partea de down
        # 2. se inregistreaz acare au fost output-urile blocurilor convolutionale in partea de down
        i = 0
        short_skip_connections = []
        for down, skrinker in zip(self.downs, self.skrinker_conv):
            short_skip_connections.append(x)
            x = down(x)
            skip_connections.append(x)

            # mecanismul de skip_connection scurt, intra-bloc
            x = torch.cat((x, short_skip_connections.pop(0)), dim = 1)

            # mecanismul de reducere a spatiului. In acest caz se va folosi shrink_conv, in loc de pooling
            x = skrinker(x)

            i += 1

        # se trece prin spatiul latent
        short_skip_connections = []
        short_skip_connections.append(x)
        x = self.bottleneck(x)
        x = torch.cat((x, short_skip_connections.pop(0)), dim=1)
        x = self.ups_expand_latent(x)

        # partea de up
        skip_connections = skip_connections[::-1] # inverseaza ordinea vectorului
        short_skip_connections = []
        first = True
        for i in range(0, len(self.ups_expand)): # se merge cu PASUL DE 2 deoarece syunt 2 convolutii
            #x = self.ups_expand[i](x)
            # artificiu, ca sa ia doar prima comp din sp latent
            #? totusi, nu cumva e lasata nefolosita o convolutie si asta va afecta GRAFUL?
            if not first:
                x = self.ups_expand[i](x) # aici este upsample-ul
            first = False
            skip_connection = skip_connections[i] # totusi, skip_connection-urile nu se parcurg cu pas de 2s
            #concatenated_x = torch.cat((skip_connection, x), dim=1)  # concatenarea x cu skip, de-a lungul axei canalelor (dim=1)
            x = torch.cat((skip_connection, x), dim=1)  # concatenarea x cu skip, de-a lungul axei canalelor (dim=1)

            ##################### MECANISMUL DE ATENTII PER CANAL #########################
            
            channedl_weights = self.squeeze_and_extraction[i](x)
            channedl_weights = F.sigmoid(channedl_weights) # il pun aici pentyru ca nu pot inainutrul blocului
            x = x * channedl_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)

            ##################### MECANISMUL DE ATENTII PER CANAL #########################

            short_skip_connections.append(x)
            x = self.ups_conv[i](x)
            x = torch.cat((short_skip_connections.pop(0), x), dim=1)

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

            #x = self.ups_expand[i+1](concatenated_x)

        x = self.adapter_layer(x)

        ######
        # FUCNTIA DE ACTIVARE
        ######
        # sigmoida se aplica daca functia de cost vrea valori intre 0 si 1. 
        x = F.sigmoid(x)

        return x

def test():
    x = torch.randn((4,4,240,240))
    model = UNET_binar_brats_2D_v1(4, 1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)

if __name__ == "__main__":
    test()


