from .mnist import MNIST
from torch.utils.data import DataLoader
import torch
from .clevr import CLEVR
from .dsprite import MultiDSprites
def make_dataloader(cfg, mode):
    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
    elif mode == 'val':
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = False
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
    
    # build dataset
    dataset = make_dataset(cfg, mode)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle, drop_last=True, num_workers=num_workers)
    
    return dataloader

# def collate_fn(batch):
#     """
#     :param batch: list, each being a tuple (data, mask)
#         data: (*), image
#         mask: (*), mask
#     :return: (data, mask), where mask is a list
#     """
#
#     data, mask = zip(*batch)
#     data = torch.stack(data, dim=0)
#
#     return data, mask
    
    

def make_dataset(cfg, mode):
    if cfg.DATASET.TRAIN == 'MNIST':
        return MNIST('data/MNIST', mode)
    elif cfg.DATASET.TRAIN == 'CLEVR':
        return CLEVR(cfg)
    elif cfg.DATASET.TRAIN == 'DSPRITES':
        return MultiDSprites(cfg)
