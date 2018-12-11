from dataloaders.DRIVE import DRIVE



def get_dataloader(dataset):

    return {
        "drive" : DRIVE
    }[dataset]
