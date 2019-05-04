import torch
from .vae import VAE
from lib.modeling.iter_net import IterNet
from lib.modeling.iodine import IODINE

def make_model(cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    device_ids = cfg.MODEL.DEVICE_IDS
    if not device_ids: device_ids = None # use all devices
    model = _make_model(cfg)
    model = model.to(device)
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model

def _make_model(cfg):
    if cfg.MODEL.NAME == 'VAE':
        return VAE(28 * 28, 128)
    elif cfg.MODEL.NAME == 'Iter':
        return IterNet(28 * 28, 64)
    elif cfg.MODEL.NAME == 'IODINE':
        return IODINE(5, 5, 128)

