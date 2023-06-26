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
import math

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


class InceptionBlock(nn.Module):
    '''
    Implementeaza clasicul Bottleneclk sau Inception block, adic un sandwich de convolutie 3x3, prinsa intre
    doua convolutii 1x1 si cu un skip conection iunchis peste ea.
    '''

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(InceptionBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # DILATAREA se petrece in CONV2
        #? sa il pun padding = dilation sau padding = same?
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # salvez reziduul
        reziduu = x

        # 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        '''
        #? Vreau sa scot asata
        if self.downsample is not None:
            reziduu = self.downsample(x)
        '''

        # se aduna reziduul (poate testez si o concatenare?)
        out += reziduu
        # pun relu abia dupa ce s-a adaugat si output-ul precedent (desi e deja FACUT RELU PE EL. pOATE E INUTIL SI IL MUT)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    '''
    Implementeaza un Resnet bazat pe un anumti tip de blocuri. (o sa transmit InceptionBlocks)

    '''

    def __init__(self, block, layers, num_classes, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = nn.BatchNorm2d
        '''
        #? scot
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        '''
        self.conv = nn.Conv2d
        '''
        #? vrerau sa o scot
        self.conv = Conv2d if weight_std else nn.Conv2d
        '''

        super(ResNet, self).__init__()
        #? DE CE EI PUN UN 7x7 LA INTRARE sau 3 straturi consecutive de 3x3x3? Stiu ca sunt aproximativ echivalente, dar dc nu pun unul singur de 3x3?
        self.conv1 = self.conv(4, 64, kernel_size=3, stride=1, padding='same',bias=False) # asta e doar de test

        '''
        # prefer sa il implementez cu un simplu stratr de convolutie
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        '''
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Urmeaza crearea blocurilor ResNEt, ca InceptionBlocks, desi s-are putrea primi orice la intrare
        #! am inclus atentiile/skip connections peste blocrui in InceptionBlock
        # o sa definesc infelxibil nr de layers. Poate o sa scrimb asata sa fac cu stiva ca la UNET
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

        # Atrous Spatial Pyramid Pooling
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, nr_blocuri, stride=1, dilation=1):
        '''
        
        '''
        # metoda care primeste un tip de bloc (InceptionBlock) si creeaza o insiruitre de nr_blocuri astfel de blocuri, tindand cont de dialtare

        downsample = None
        # Aici practic e un adaptor. nu se face downsaple decat daca stride nu e 1 sau exista dilatare sau daca expanisunea nu s-ar face corect
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, nr_blocuri):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers) # operatorul * face unpack => mai multe argumente, pentru ca asa vrea constructoul nn.Sequential sa primeasca

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x



class DeepLabv3_v1(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        '''
        :param in_channels: number of input chanels in the UNET
        :param out_channels: number of classes OR 1, in case of a binary classification        '''
        super(DeepLabv3_v1, self).__init__()

    
    def forward(self, x):
        

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
    model = DeepLabv3_v1(4, 1)
    preds = model(x)
    #print(x.shape)
    #print(preds.shape)

if __name__ == "__main__":
    test()




















