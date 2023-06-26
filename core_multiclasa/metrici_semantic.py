"""
EXISAT 2 VARIANTE:




VARIANTA 1:

'''
    a)
        Calculul fucntiei de coist se face in raport cu: ED, ET, NCR.

        se presupun canalele:
        la PREDICTIE (Out retea = pred, aici):
        ch 0 = WT
        ch 1 = TC
        ch 2 = ET

        Valorile
        La TARGET:
        value 1 = NCR
        value 4 = ET
        value 2 = ED
    b) sigmoida
'''

VARIANTA 2:



'''
    a) 
        Calculul fucntiei de coist se face in raport cu: ED, ET, NCR.

        se presupun canalele:
        la PREDICTIE (Out retea = pred, aici):
        ch 0 = NCR
        ch 1 = ET
        ch 2 = ED

        Valorile
        La TARGET:
        value 1 = NCR
        value 4 = ET
        value 2 = ED
    b) softmax
'''
# COD:

 # Extrag GT-urile celor 3 canale:
        target_NCR = np.where(pred[:, :] == 1, 1, 0) # 1 este label-ul NCR
        
        target_WT = np.where(pred[:, :] != 0, 1, 0) # toate label-urile nenume reprezinta WT

        # Denumesc cele 2 canale extrase

"""




import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import cv2
from torchgeometry.losses.one_hot import one_hot
import scipy



# Variabilele ponderi globale, calculate poe intregul set de date

# befotre softmax
global_W_ED = 2.108 
global_W_ET = 2.539 
global_W_NCR = 7.583
#global_W_NCR = 0

'''
global_W_ED = 1 + 0.1098
global_W_ET = 1
global_W_NCR = 4
'''


#################### dice_loss ###########################

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(pred, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pred):
            raise TypeError("pred type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not pred.device == target.device:
            raise ValueError(
                "pred and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))
        
        pred_soft = pred # o face reteaua
    
        intersection = torch.sum(pred_soft * target) # se pace suma tutuorr elemntelor din MATRICEA produs element cu element dintre predictie si target
        cardinality = torch.sum(pred_soft + target) # suma tuturor elementelor din suma elem cu elem dintre pred si target
        '''
        intersection = torch.sum(pred_soft * target, dims)
        cardinality = torch.sum(pred_soft + target, dims)
        '''
        dice_score = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score)


######################
# functional interface
######################


def dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(pred, target)


#################### dice_loss ###########################

#################### weighted_dice_loss ###########################

class GeneralisedDiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(pred, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(GeneralisedDiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        '''
        Calculul fucntiei de coist se face in raport cu: ED, ET, NCR.

        se presupun canalele:
        la PREDICTIE (Out retea = pred, aici):
        ch 0 = NCR
        ch 1 = ET
        ch 2 = ED
        ch 3 = backgorund

        Valorile
        La TARGET:
        value 1 = NCR
        value 4 = ET
        value 2 = ED
        '''



        if not torch.is_tensor(pred):
            raise TypeError("pred type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not pred.device == target.device:
            raise ValueError(
                "pred and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))
        
        pred_sigm = pred # se face deja sigmoida, la output-ul retelei. Este inclus in arhitectura
        
        ##print(target.shape, pred.shape)

        # Extrag GT-urile celor 3 canale:
        target_ET = torch.where(target[:, 0, :, :] == 255, 1, 0) # 4 este label-ul ET
        target_NCR = torch.where(target[:, 0, :, :] == 63, 1, 0)
        target_ED = torch.where(target[:, 0, :, :] == 127, 1, 0) # toate label-urile nenume reprezinta WT
        target_background = torch.where(target[:, 0, :, :] == 0, 1, 0) # toate label-urile nenume reprezinta WT

        ##print(np.unique(target_ET.cpu().data.numpy()))
        ##print(np.unique(target_NCR.cpu().data.numpy()))
        ##print(np.unique(target_ED.cpu().data.numpy()))

        # VARIANTA 2

        # binarizare si regula de decizie. pentru o pozitie de voxel, se face 1 canalul activarii maxime si 0 restuil canalelor
        # are sens ssa fac binarizarewa?
        #pred = F.one_hot(pred)

        # Creez hartile asociate OUT-ului:
        predicted_NCR = pred[:,0,:,:]
        predicted_ET = pred[:,1,:,:]
        predicted_ED = pred[:,2,:,:]
        #predicted_background = pred[:,3,:,:]

        ##print(target_ET.shape, predicted_ET.shape)

        # SALV&VIEW $$$$$$$$$
        
        #target1 = (target_ED.cpu().data.numpy())[0, 0, :, :]
        #out1 = (predicted_ED.cpu().data.numpy())[0, :, :]
        ##print(target.shape)
        ##print('[aidshf9asdg9f087ugaews9fgv9asgfawsgf]')
        #cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\ET.png', np.uint8(255*target_ET[0,:,:].cpu().data.numpy()))
        #cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\ED.png', np.uint8(255*target_ED[0,:,:].cpu().data.numpy()))
        #cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\NCR.png', np.uint8(255*target_NCR[0,:,:].cpu().data.numpy()))
        #cv2.imwrite(r'E:\an_4_LICENTA\Workspace\junkdata\GT.png', np.uint8(target[0,0,:,:].cpu().data.numpy()))
        #afs = input()
        
        # SALV&VIEW $$$$$$$$$

        #### 
        # VARIANTA in care ponderile claselor se calculaeaza pe intregul set de date
        ####
        # varianta cu normalizare
        '''
        W_ET = global_W_ET/(global_W_ET + global_W_NCR + global_W_ED)
        W_NCR = global_W_NCR/(global_W_ET + global_W_NCR + global_W_ED)
        W_ED = global_W_ED/(global_W_ET + global_W_NCR + global_W_ED)
        '''
        # varianta cu softmax
        '''
        W_ED, W_ET, W_NCR = scipy.special.softmax([W_ED, W_ET, W_NCR])
        '''


        #### 
        # VARIANTA in care ponderile claselor se calculaeaza per batch
        ####
        
        # se determina ponderile asociate fiecarei clase. Se calculeaza in raport cu gt-ul "target"
        '''
        if torch.count_nonzero(target_ET) == 0:
            count_ET = 0
        else:
            count_ET = torch.count_nonzero(target_ET)
        if torch.count_nonzero(target_NCR) == 0:
            count_NCR = 0
        else:
            count_NCR = torch.count_nonzero(target_NCR)
        if torch.count_nonzero(target_ED) == 0:
            count_ED = 0
        else:
            count_ED = torch.count_nonzero(target_ED)
        '''
        
        count_ET = torch.count_nonzero(target_ET)
        count_NCR = torch.count_nonzero(target_NCR)
        count_ED = torch.count_nonzero(target_ED)
        

        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        #W_ET = count_ET/(count_ET + count_NCR + count_ED)
        #W_NCR = count_NCR/(count_ET + count_NCR + count_ED)
        #W_ED = count_ED/(count_ET + count_NCR + count_ED)

        # Calculul ponderior, nenormalizare
        if count_ET == 0:
            W_ET_nenorm = 0
        else:
            W_ET_nenorm = (count_ET + count_NCR + count_ED)/count_ET

        if count_NCR == 0:
            W_NCR_nenorm = 0
        else:
            W_NCR_nenorm = (count_ET + count_NCR + count_ED)/count_NCR

        if count_ED == 0:
            W_ED_nenorm = 0
        else:
            W_ED_nenorm = (count_ET + count_NCR + count_ED)/count_ED
        
        # normalizarea ponderilor (mi se parea ca softmax da numere prea extreme)
        # acum vreau sa fie toate intre 0 si 1 deci facd operatia: (mi se parea ca softmax da numere prea extreme)
        #print(W_ET, W_NCR, W_ED)
        W_ET = W_ET_nenorm/(W_ET_nenorm + W_NCR_nenorm + W_ED_nenorm)
        W_NCR = W_NCR_nenorm/(W_ET_nenorm + W_NCR_nenorm + W_ED_nenorm)
        W_ED = W_ED_nenorm/(W_ET_nenorm + W_NCR_nenorm + W_ED_nenorm)
        #print(W_ET, W_NCR, W_ED)

        #print('llll')
        #print(W_ET, W_NCR, W_ED)
        #W_ET, W_NCR, W_ED = scipy.special.softmax([W_ET.cpu(), W_NCR.cpu(), W_ED.cpu()])
        #print(W_ET, W_NCR, W_ED)
        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        
        #W_ED, W_ET, W_NCR = scipy.special.softmax([W_ED.cpu(), W_ET.cpu(), W_NCR.cpu()])
        
        '''
        if torch.count_nonzero(target_background) == 0:
            W_background= 0
        else:
            W_background = 1/torch.count_nonzero(target_background)
        '''

        # varianta a) in care se tine cont si de background
        '''
        numarator = W_NCR * torch.sum(predicted_NCR * target_NCR) + W_ET * torch.sum(predicted_ET * target_ET) + W_ED * torch.sum(predicted_ED * target_ED) + W_background * torch.sum(predicted_background * target_background)
        numitor = W_NCR * torch.sum(predicted_NCR + target_NCR) + W_ET * torch.sum(predicted_ET + target_ET) + W_ED * torch.sum(predicted_ED + target_ED) + W_background * torch.sum(predicted_background + target_background)
        '''
        # varianta b) in care nu se tine cont de background
        '''
        numarator = W_NCR * torch.sum(predicted_NCR * target_NCR) + W_ET * torch.sum(predicted_ET * target_ET) + W_ED * torch.sum(predicted_ED * target_ED)
        numitor = W_NCR * torch.sum(predicted_NCR + target_NCR) + W_ET * torch.sum(predicted_ET + target_ET) + W_ED * torch.sum(predicted_ED + target_ED)
        generalised_dice_score = 2*numarator/numitor
        '''
        # varianta c) in care NU folosesc formula din apepr GDL, ci ponderez fiecare dice loss in parter:
        
        DS_NCR = 1 - dice_loss(predicted_NCR, target_NCR)
        DS_ET = 1 - dice_loss(predicted_ET, target_ET)
        DS_ED = 1 - dice_loss(predicted_ED, target_ED)

        generalised_dice_score = W_NCR * DS_NCR + W_ET * DS_ET + W_ED * DS_ED
        
        
        return torch.mean(1. - generalised_dice_score)


######################
# functional interface
######################


def weighted_dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return GeneralisedDiceLoss()(pred, target)


#################### weighted_dice_loss ###########################




#################### weighted_dice_loss ###########################

class My_cross_entropy(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(pred, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(My_cross_entropy, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        '''
        Calculul fucntiei de coist se face in raport cu: ED, ET, NCR.

        se presupun canalele:
        la PREDICTIE (Out retea = pred, aici):
        ch 0 = NCR
        ch 1 = ET
        ch 2 = ED
        ch 3 = backgorund

        Valorile
        La TARGET:
        value 1 = NCR
        value 4 = ET
        value 2 = ED
        '''



        if not torch.is_tensor(pred):
            raise TypeError("pred type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not pred.device == target.device:
            raise ValueError(
                "pred and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))
        
        pred_sigm = pred # se face deja sigmoida, la output-ul retelei. Este inclus in arhitectura
        
        ##print(target.shape, pred.shape)

        # Extrag GT-urile celor 3 canale:
        target_ET = torch.where(target[:, 0, :, :] == 255, 1, 0) # 4 este label-ul ET
        target_NCR = torch.where(target[:, 0, :, :] == 63, 1, 0)
        target_ED = torch.where(target[:, 0, :, :] == 127, 1, 0) # toate label-urile nenume reprezinta WT
        target_background = torch.where(target[:, 0, :, :] == 0, 1, 0) # toate label-urile nenume reprezinta WT

        ##print(np.unique(target_ET.cpu().data.numpy()))
        ##print(np.unique(target_NCR.cpu().data.numpy()))
        ##print(np.unique(target_ED.cpu().data.numpy()))

        # VARIANTA 2

        # binarizare si regula de decizie. pentru o pozitie de voxel, se face 1 canalul activarii maxime si 0 restuil canalelor
        # are sens ssa fac binarizarewa?
        #pred = F.one_hot(pred)

        # Creez hartile asociate OUT-ului:
        predicted_NCR = pred[:,0,:,:]
        predicted_ET = pred[:,1,:,:]
        predicted_ED = pred[:,2,:,:]
        #predicted_background = pred[:,3,:,:]

        #### 
        # VARIANTA in care ponderile claselor se calculaeaza per batch
        ####
        
        # se determina ponderile asociate fiecarei clase. Se calculeaza in raport cu gt-ul "target"
        if torch.count_nonzero(target_ET) == 0:
            count_ET = 0
        else:
            count_ET = torch.count_nonzero(target_ET)
        if torch.count_nonzero(target_NCR) == 0:
            count_NCR = 0
        else:
            count_NCR = torch.count_nonzero(target_NCR)
        if torch.count_nonzero(target_ED) == 0:
            count_ED = 0
        else:
            count_ED = torch.count_nonzero(target_ED)
        W_ET = count_ET/(count_ET + count_NCR + count_ED)
        W_NCR = count_NCR/(count_ET + count_NCR + count_ED)
        W_ED = count_ED/(count_ET + count_NCR + count_ED)

        #### 
        # VARIANTA in care ponderile claselor se calculaeaza pe intregul set de date
        ####
        # varianta cu normalizare
        '''
        W_ET = global_W_ET/(global_W_ET + global_W_NCR + global_W_ED)
        W_NCR = global_W_NCR/(global_W_ET + global_W_NCR + global_W_ED)
        W_ED = global_W_ED/(global_W_ET + global_W_NCR + global_W_ED)
        '''
        #W_ET = 1
        #W_NCR = 1
        #W_ED = 1
        
        

        # varianta a) in care se tine cont si de background
        '''
        numarator = W_NCR * torch.sum(predicted_NCR * target_NCR) + W_ET * torch.sum(predicted_ET * target_ET) + W_ED * torch.sum(predicted_ED * target_ED) + W_background * torch.sum(predicted_background * target_background)
        numitor = W_NCR * torch.sum(predicted_NCR + target_NCR) + W_ET * torch.sum(predicted_ET + target_ET) + W_ED * torch.sum(predicted_ED + target_ED) + W_background * torch.sum(predicted_background + target_background)
        '''
        # varianta b) in care nu se tine cont de background
        '''
        numarator = W_NCR * torch.sum(predicted_NCR * target_NCR) + W_ET * torch.sum(predicted_ET * target_ET) + W_ED * torch.sum(predicted_ED * target_ED)
        numitor = W_NCR * torch.sum(predicted_NCR + target_NCR) + W_ET * torch.sum(predicted_ET + target_ET) + W_ED * torch.sum(predicted_ED + target_ED)
        generalised_dice_score = 2*numarator/numitor
        '''
        # varianta c) in care NU folosesc formula din apepr GDL, ci ponderez fiecare dice loss in parter:
        DS_NCR = F.cross_entropy(predicted_NCR.double(), target_NCR.double())
        DS_ET = F.cross_entropy(predicted_ET.double(), target_ET.double())
        DS_ED = F.cross_entropy(predicted_ED.double(), target_ED.double())

        CE = W_NCR * DS_NCR + W_ET * DS_ET + W_ED * DS_ED
        
        return CE


######################
# functional interface
######################


def my_cross_entropy(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return My_cross_entropy()(pred, target)


#################### my_cross_entropy ###########################




#################### dice_loss tresholded #########################

class DiceLoss_tresholded(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    """

    def __init__(self) -> None:
        super(DiceLoss_tresholded, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pred):
            raise TypeError("pred type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not pred.device == target.device:
            raise ValueError(
                "pred and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))
        
        pred = (pred>0.5).float()
    
        intersection = torch.sum(pred * target) # se pace suma tutuorr elemntelor din MATRICEA produs element cu element dintre predictie si target
        cardinality = torch.sum(pred + target) # suma tuturor elementelor din suma elem cu elem dintre pred si target
        '''
        intersection = torch.sum(pred_soft * target, dims)
        cardinality = torch.sum(pred_soft + target, dims)
        '''
        dice_score_tresholded = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score_tresholded)


######################
# functional interface
######################


def dice_loss_tresholded(
        pred: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss_tresholded()(pred, target)


#################### dice_loss tresholded #########################








