from torch.optim import Adam
from torch.optim import SGD
from  torch.optim.lr_scheduler import StepLR, MultiStepLR

def get_optimizer(optimizer):

    return {
        "adam": Adam,
        "sgd": SGD
    }[optimizer]


def get_lr_scheduler(scheduler_type):

    return {
        'step_lr' : StepLR,
        'multi_step_lr': MultiStepLR
    }[scheduler_type]