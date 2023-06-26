
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from torchgeometry.losses.one_hot import one_hot


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
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        
        input_soft = input
    
        intersection = torch.sum(input_soft * target) # se pace suma tutuorr elemntelor din MATRICEA produs element cu element dintre predictie si target
        cardinality = torch.sum(input_soft + target) # suma tuturor elementelor din suma elem cu elem dintre pred si target
        '''
        intersection = torch.sum(input_soft * target, dims)
        cardinality = torch.sum(input_soft + target, dims)
        '''
        dice_score = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score)


######################
# functional interface
######################


def dice_loss(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss()(input, target)


#################### dice_loss ###########################


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
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        
        input = (input>0.5).float()
    
        intersection = torch.sum(input * target) # se pace suma tutuorr elemntelor din MATRICEA produs element cu element dintre predictie si target
        cardinality = torch.sum(input + target) # suma tuturor elementelor din suma elem cu elem dintre pred si target
        '''
        intersection = torch.sum(input_soft * target, dims)
        cardinality = torch.sum(input_soft + target, dims)
        '''
        dice_score_tresholded = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score_tresholded)


######################
# functional interface
######################


def dice_loss_tresholded(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss_tresholded()(input, target)


#################### dice_loss tresholded #########################








