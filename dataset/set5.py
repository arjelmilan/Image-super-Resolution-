
import glob
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

def make_dataset(dir):
  images = glob.glob(f'{dir}/*')
  return images

def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return v2.Compose([
        v2.ToTensor(),
    ])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class TestDataset(Dataset):
    def __init__(self, hr_dir, lr_dir,scale):
        super().__init__()
        self.dir_lr = lr_dir
        self.dir_hr = hr_dir
        self.scale = scale
        self.images_hr, self.images_lr = self._scan()

    def __getitem__(self, index):
        lr, hr = self._load_file(index)
        # lr, hr = set_channel(lr, hr, n_channels=3)
        input = np2tensor()(lr)
        target = np2tensor()(img_modcrop(hr, self.scale))

        return input, target

    def __len__(self):
        return len(self.images_hr)

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = idx
        lr = load_image(self.images_lr[idx])
        hr = load_image(self.images_hr[idx])
        return lr, hr

