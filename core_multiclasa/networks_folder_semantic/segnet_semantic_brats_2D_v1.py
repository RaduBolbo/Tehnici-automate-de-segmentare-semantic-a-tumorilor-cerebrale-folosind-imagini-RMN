'''
This is a SegNet

Comparativ cu UNet-ul, se vor scoate skip connection-urile dintre codor si decodor si, evident, mecanismul de atenii si, in plus, 
in loc de featuire map-uri, intre codor si de codor, se vor transmite indicii de pooling

V1: 

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
    def __init__(self): # ! nu mai este nevoie de transmiterea argumentului de feature, deoarecve maxpool opereaza in virtutea dimensiunii kernelului si nu poate modifica oricum numarul de feature-s, spre de osebire de o convolutie
        super(Shrinker, self).__init__()
        self.maxpool = nn.Sequential(
            # SE INLOCUIESTE CONVOLUTIA cu stride 2
            #nn.Conv2d(feature*2, feature*2, kernel_size=2, stride=2),
            # se intorduce maxpool
            nn.MaxPool2d((2,2), stride=None, padding=0, dilation=1, return_indices=True)
            # NU mai e nevoie de BatchNorm si nici de f. de activare daca nu mai fac convolutie, deoarece nu se produc valori noi in urma Unpool
            #nn.BatchNorm2d(feature*2),
            #nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.maxpool(x) # ! aici se vor returna 2 entitati: tensorul pooled si indicii

class Upsample_block(nn.Module):
    def __init__(self): 
        super(Upsample_block, self).__init__()
        self.Upsample = nn.Sequential(
            # Se scoate nn.Upsample si se va inlocui cu .nn.MaxUnpool2d
            #nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # Se scoate mo0pmemmnatn si convolutia
            #nn.Conv2d(6*feature, feature, kernel_size=3, stride=1, padding = 'same')
            nn.MaxUnpool2d((2,2), stride=None, padding='same') # e ok padding same?
            )
    def forward(self, x, indices): # este nevoie de indici, la forward
        return self.Upsample(x, indices, x.size())
        
# 
############ SCHIMBARE DE PLAN: NU POT INCLUDE IN ..Sequential un nn.MaxUnpool2d si sa ii transmit si indicvii prin metoda forward()
#
class Upsample_latent(nn.Module):
    def __init__(self):
        super(Upsample_latent, self).__init__()
        self.Upsample = nn.Sequential(
            # Se scoate Upsample
            # nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            # Momenatn se scoate si convolutia
            #nn.Conv2d(4*feature, feature, kernel_size=3, stride=1, padding = 'same')
            nn.MaxUnpool2d((2,2), stride=None, padding='same') # e ok padding same?
            )
    def forward(self, x, indices):
        return self.Upsample(x, indices, x.size())


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

class FeatureReductor(nn.Module):
    def __init__(self, features):
        super(FeatureReductor, self).__init__()
        self.conv = nn.Sequential( # e un CONTAINER SECVENTIAL, ce va contine straturi
            nn.Conv2d(4*features, features, 3, 1, bias=False, padding = 'same'), # bias-ul asociat va fi oricum eliminat de batch norm, deci e nenecesar
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Segnet_semantic_brats_v1(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features = [64, 128, 256, ]):
        '''
        :param in_channels: number of input chanels in the UNET
        :param out_channels: number of classes OR 1, in case of a binary classification
        :param features: the numer of channels specific to each convolutional block
        '''
        super(Segnet_semantic_brats_v1, self).__init__()
        self.ups_expand = nn.ModuleList()
        self.ups_conv = nn.ModuleList()
        self.downs = nn.ModuleList()
        #self.squeeze_and_extraction = nn.ModuleList()
        #self.downs_conv = nn.ModuleList()
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.skrinker = nn.ModuleList() # in loc de maxpooling
        self.ups_featurereduction_conv = nn.ModuleList()

        # pentru stratul d einput:
        self.input_layer = Input_layer(in_channels, features[0])
        #self.input_layer = Input_layer(in_channels, 8)

        # se va crea UNET-ul:

        # Down part
        in_channels_for_downs = features[0] # asta deoarece
        for feature in features:
            self.downs.append(DoubleConv(feature, feature)) # inputul etse in_channels, iar la output e cat specifica feature
            self.skrinker.append(Shrinker()) # se merge pe ideea ca merg din 2 in 2

        # Up part
        for feature in reversed(features):
            # partea de upsampling -> o fac, in cazul asta, prind dublarea dimensionalitatii cu o ConvTranspinse2d
            self.ups_expand.append(
                #Upsample_block()
                #nn.MaxUnpool2d((2,2), stride=None, padding='same')
                nn.MaxUnpool2d((2,2), stride=None, padding=0)
            )
            #
           # self.ups.append(DoubleConv(feature*2, feature))


        for feature in reversed(features):
            self.ups_conv.append(DoubleConv(feature, feature))

        for feature in reversed(features):
            self.ups_featurereduction_conv.append(FeatureReductor(feature))

        # spatiul latent
        self.bottleneck = DoubleConv(features[-1]*2, features[-1]*2)
        self.bottleneck_reducing = nn.Conv2d(4*features[-1], features[-1], 3, 1, bias=False, padding = 'same')

        ############## ADAUGARE ################
        # upsample-ul din spatiul latent
        #self.ups_expand_latent = nn.MaxUnpool2d((2,2), stride=None, padding='same') # e un bug in pytorch care nu ma lasa sa folosesc upsample oricum
        self.ups_expand_latent = nn.MaxUnpool2d((2,2), stride=None, padding=0)
        ############## ADAUGARE ################

        # adaptorul la numarul de clase
        self.adapter_layer = nn.Conv2d(features[0]*2, out_channels, kernel_size=1, padding = 'same')

        #self.testare = nn.Conv2d(4, 8, kernel_size=2, stride=1, padding = 'same')

    def forward(self, x):
        # SE SCOT skip connection-urile, la SegNet
        #skip_connections = []

        # Prima data se trece prin layer-ul de input, care sa mor daca stiu de ce e 2x2:
        x = self.input_layer(x)
        #x = self.testare(x)

        # 1. se trece prin partea de down
        # 2. se inregistreaz acare au fost output-urile blocurilor convolutionale in partea de down
        i = 0
        short_skip_connections = []
        indices_list = []
        for down, skrinker in zip(self.downs, self.skrinker):
            # las partea asta de short skip connection si in SegNet
            short_skip_connections.append(x)
            x = down(x)
            ##print(x.shape, '3')
            #skip_connections.append(x) # SegNet => scot Skip Conn
            

            # mecanismul de skip_connection scurt, intra-bloc
            # Pe asta il las si la SegNet
            x = torch.cat((x, short_skip_connections.pop(0)), dim = 1)
            ##print(x.shape, '4')

            # mecanismul de reducere a spatiului. In acest caz se va folosi shrink_conv, in loc de pooling
            x, indici = skrinker(x)
            ##print(x.shape, '5')
            ##print('indici: ', indici.shape)
            indices_list.append(indici)

            i += 1
        # se trece prin spatiul latent
        short_skip_connections = []
        short_skip_connections.append(x)
        x = self.bottleneck(x)
        ##print(x.shape, '6')
        x = torch.cat((x, short_skip_connections.pop(0)), dim=1)
        ##print(x.shape, '7')

        indices_list = indices_list[::-1] # inverseaza ordinea vectorului
        #x = self.ups_expand_latent(x, indices_list[0])
        x = self.ups_expand_latent(x, torch.cat((indices_list[0], indices_list[0]), dim=1)) # Trebuie replicat acelasi lucru
        ##print(x.shape, '8')
        # aici mai trebuie si o convolutie pentru reudcerea numarului de featue maps

        # partea de up
        #skip_connections = skip_connections[::-1] # inverseaza ordinea vectorului
        
        short_skip_connections = []
        first = True
        for i in range(0, len(self.ups_expand)): # se merge cu PASUL DE 2 deoarece syunt 2 convolutii
            #x = self.ups_expand[i](x)
            #skip_connection = skip_connections[i] # se scoate la SegNet
            indici = indices_list[i]
            if not first:
                #x = self.ups_expand[i](x, indici) # aici este upsample-ul
                x = self.ups_expand[i](x, torch.cat((indices_list[i], indices_list[i]), dim=1)) # aici este upsample-ul
                ##print(x.shape, '9')
                
            first = False
            
            #concatenated_x = torch.cat((skip_connection, x), dim=1)  # concatenarea x cu skip, de-a lungul axei canalelor (dim=1)
            # NU se mai face concatenarea cu feature-urile de la codor
            #x = torch.cat((skip_connection, x), dim=1)  # concatenarea x cu skip, de-a lungul axei canalelor (dim=1)

            ##################### MECANISMUL DE ATENTII PER CANAL SE SCOATE #########################
            
            #channedl_weights = self.squeeze_and_extraction[i](x) # nu mai exisat bloc de atentii
            #channedl_weights = F.sigmoid(channedl_weights) # il pun aici pentyru ca nu pot inainutrul blocului
            #x = x * channedl_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)

            ##################### MECANISMUL DE ATENTII PER CANAL SE SCOATE #########################
            x = self.ups_featurereduction_conv[i](x) # convolutia din convolve + expand
            ##print(x.shape, '10')
            
            short_skip_connections.append(x)
            x = self.ups_conv[i](x)
            ##print(x.shape, '11')
            x = torch.cat((short_skip_connections.pop(0), x), dim=1)
            ##print(x.shape, '12')

            # PROTECTIE
            # ? nu e o problema? Nu sparge graful computational?
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
        # !!!! sigmoida se va aplica deoarece output-urile NU sunt mutual exclusive: am WT pe canalul 0, care inglobeaza ET si NCR !!!!
        x = F.sigmoid(x) # SIGMOIDA O FOLOSESC PT. CA LA LOSS NU EXISTA INTERSCHIMABRE DE INFORMATIE INTRE CANALE. FIECARE CANAL ARE CATE DOUA CLASE SI ARE DICE SCORE-UL LUI SI F. ACTIVARE LOGISTICA PROPRIE. NU SE FORMEAZA SISTEKM COMPLET DE EVENMIMENTE
        #x = F.softmax(x) # multiclasa pt clase DISJUNCTE: NCR ET ED

        return x

def test():
    x = torch.randn((4,4,240,240))
    model = Segnet_semantic_brats_v1(4, 1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)

if __name__ == "__main__":
    test()




















