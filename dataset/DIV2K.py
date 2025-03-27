from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import cv2
import os
from copy import deepcopy
from common import get_patch,augment

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]



IMG_EXTENSIONS = [
    '.png', '.npy',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)





def np2Tensor(*args, rgb_range=1):
      def _np2Tensor(img):
          np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
          tensor = torch.from_numpy(np_transpose.copy()).float()
          tensor.mul_(rgb_range / 255)

          return tensor

      return [_np2Tensor(a) for a in args]


class div2k(Dataset):
    def __init__(self,hr_dir,lr_dir ,  patch_size, scale,train):
        self.dir_lr = lr_dir
        self.dir_hr = hr_dir
        self.patch_size = patch_size
        self.train = train
        self.scale = scale
        
        self.data = self.make_dataset()

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        # lr, hr = set_channel(lr, hr, n_channels=3)
        lr_tensor, hr_tensor = np2Tensor(lr, hr, rgb_range=1)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.data)


    def _get_patch(self, img_in, img_tar):
        patch_size = self.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar


    def make_dataset(self):
            if self.train:
                lr_files = os.listdir(self.dir_lr)
            
                paired_data = []
                for lr_file in lr_files:
                    tmp = deepcopy(lr_file)
                    paired_data.append((lr_file, tmp.replace(f'x{4}.png', '.png')))
                    data = paired_data
            else:
                lr_files = os.listdir(self.dir_lr)
                lr_files = lr_files[:10]
                paired_data = []
                for lr_file in lr_files:
                    tmp = deepcopy(lr_file)
                    paired_data.append((lr_file, tmp.replace(f'x{4}.png', '.png')))
                    data = paired_data                
                
                
            return data


    def _load_file(self, idx):
        lr_name, hr_name =self.data[idx]
        lr_path = os.path.join(self.dir_lr, lr_name)
        hr_path = os.path.join(self.dir_hr, hr_name)
        lr = default_loader(lr_path)
        hr = default_loader(hr_path)
        return lr, hr
