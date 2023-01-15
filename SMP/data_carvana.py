"""
Functions to work with carvana dataset.

carvana_dir/
    images/
    masks/
"""

from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import os


class Dataset(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0]+"_mask.gif") for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = dict(
            car=255,
        )
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def _read_gif(self, file_path: str) -> cv2.Mat:
        cap = cv2.VideoCapture(file_path)
        ret, image = cap.read()
        cap.release()

        if ret:
            return image
        return None
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._read_gif(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values.values()]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
