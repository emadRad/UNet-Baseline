import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    '''
    Dice Loss

    '''
    def __init__(self, weights=None):
        super(DiceLoss, self).__init__()
        self.weights = weights

    def forward(self, output, target, ignore_index=None):
        '''
        Inputs:
            -- output: N x C x H x W Variable
            -- target : N x C x W LongTensor with starting class at 0
            -- weights: C FloatTensor with class wise weights
            -- ignore_index: ignore index from the loss
        '''

        eps = 0.001
        output = F.softmax(output, dim=1)

        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if self.weights is None:
            self.weights = 1

        intersection = output * encoded_target
        numerator = (2 * intersection.sum(0).sum(1).sum(1)) + eps
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = self.weights * (1 - (numerator / denominator))  # Channel-wise weights

        return loss_per_channel.sum() / output.size(1)



class CrossEntropy2D(nn.Module):
    '''
    2D Cross-entropy loss implemented as negative log likelihood
    '''

    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy2D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)



class CombinedLoss(nn.Module):
    '''
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    '''

    def __init__(self, weight_dice=None, weight_ce=None):
        super(CombinedLoss, self).__init__()

        self.cross_entropy_loss = CrossEntropy2D(weight_ce)
        self.dice_loss = DiceLoss(weight_dice)
        # self.weight_dice = weight_dice
        # self.weight_ce = weight_ce


    def forward(self, inputx, target, weight=None):
        target = target.type(torch.LongTensor)  # Typecast to long tensor
        if inputx.is_cuda:
            target = target.cuda()

        # input_soft = F.softmax(inputx, dim=1)  # Along Class Dimension
        if weight is None:
            weight = 1

        dice_val = torch.mean(self.dice_loss(inputx, target))

        ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))

        total_loss = torch.add(dice_val, ce_val)

        return total_loss, dice_val, ce_val



key2loss = {
    'cross_entropy': CrossEntropy2D,
    'dice_loss': DiceLoss,
    'combined_loss': CombinedLoss
}


def get_loss_function(loss_info):
    if len(loss_info)==2:
        loss_name, weight = loss_info
        return key2loss[loss_name](**weight)
    else:
        loss_name = loss_info
        return key2loss[loss_name]()

