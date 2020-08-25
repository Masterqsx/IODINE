import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py
import lmdb
import logging
import pickle

class MultiDSprites(Dataset):
    def __init__(self, cfg):
        self.multi_dsprites_dir = cfg.DATASET.DSPRITES_DEEPMIND_DIR
        with open(os.path.join(self.multi_dsprites_dir, cfg.DATASET.TYPE), "rb") as f:
            self.scene_props = pickle.load(f)
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        self.fixed_mask_len = cfg.DATASET.FIXED_LEN
        self.env = lmdb.open(os.path.join(self.multi_dsprites_dir, "multi_dsprites_deepmind_on_gray_lmdb"),
                             max_readers=16, readonly=True, lock=False)
        if not self.env:
            logging.error("Cannot open lmdb on : " + os.path.join(self.multi_dsprites_dir, "multi_dsprites_deepmind_on_gray_lmdb"))
            raise FileNotFoundError()
        self.has_bbox = cfg.DATASET.HAS_BBOX

    def __del__(self):
        try:
            self.env.close()
        except:
            logging.warning("lmdb can not be closed")

    def __len__(self):
        return len(self.scene_props)

    def __getitem__(self, idx):
        object_props = self.scene_props[idx]
        image_id = object_props[0]["image_id"]
        with self.env.begin(write=False) as txn:
            img = Image.frombytes('RGB', (64, 64), txn.get(("image" + str(image_id)).encode()))
            if self.img_transform is not None:
                img = self.img_transform(img)
            masks = []
            if self.has_bbox:
                bboxs = []
            for object_idx in range(len(object_props)):
                if self.has_bbox:
                    bboxs.append(object_props[object_idx]["bounding_box"])
                obj_id = object_props[object_idx]["object_id"]
                mask = Image.frombytes('L', (64, 64), txn.get(("mask" + str(image_id) + "_" + str(obj_id)).encode()))
                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)
                masks.append(mask[0])
            # Add dummy mask for ARI metrics as pytorch tensor
            # Must be used with ToTensor in pytorch transform
            if self.fixed_mask_len is not None and self.fixed_mask_len > len(object_props):
                for dummy_mask_idx in range(len(object_props), self.fixed_mask_len):
                    masks.append(torch.zeros(64, 64))
                    if self.has_bbox:
                        bboxs.append((-1, -1, -1, -1))
        if self.has_bbox:
            return img, masks, bboxs
        return img, masks

    def collate_fn(self, batch):
        img_tensor_list = []
        masks_list = []
        if self.has_bbox:
            bboxs_list = []
            for img_tensor, masks, bboxs in batch:
                img_tensor_list.append(img_tensor)
                masks_list.append(torch.stack(masks))
                bboxs_list.append(bboxs)
            return torch.stack(img_tensor_list), masks_list, bboxs_list

        for img_tensor, masks in batch:
            img_tensor_list.append(img_tensor)
            masks_list.append(torch.stack(masks))
        return torch.stack(img_tensor_list), masks_list
        

    
    
    
    
