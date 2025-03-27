import torch
import torch.amp
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR # LR warmup
import dataset.common as common
from dataset.DIV2K import div2k
from dataset.set5 import CustomTestDataset
from torch.utils.data import DataLoader, random_split
from model_train import train
from model.model import DAT
from dataset.matlab_resize import imresize
from tqdm.auto import tqdm
import torch.nn as nn
from utilities.utils import save_model

SCALE = 2
PATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_hr = "/kaggle/input/df2kdata/DF2K_train_HR"
train_lr = "/kaggle/input/df2kdata/DF2K_train_LR_bicubic/X4"

test_hr = "/kaggle/input/df2kdata/DF2K_valid_HR"
test_lr = "/kaggle/input/df2kdata/DF2K_valid_LR_bicubic/X4"


train_dataset = div2k(train_hr,train_lr , patch_size=PATCH_SIZE, scale=SCALE,train = True )
test_dataset = div2k(test_hr,test_lr , patch_size=PATCH_SIZE, scale=SCALE,train = False )

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True,drop_last=True,num_workers = 4,pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers = 4 ,pin_memory = True)

upscale = 4
height = 64
width = 64
net = DAT(
        upscale=upscale,
        in_chans=3,
        img_size=64,
        img_range=1.,
        depth=[4,4,4,4],
        embed_dim=60,
        num_heads=[4,4,4,4],
        expansion_factor=3,
        resi_connection='1conv',
        split_size=[8,16],
                ).to(device)
model = nn.DataParallel(net, device_ids=[0, 1])

from torch.optim.lr_scheduler import StepLR,LambdaLR

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# milestones = [100,170,230,280,320,350]
milestones = [50,100,140,160,200]

loss_fn = nn.L1Loss().to(device)



epochs = 201
results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                milestones = milestones,
                # checkpoint_path = "/kaggle/input/dat-new-400/pytorch/default/1/DAT_lr5e-4_738k_milestone400.pth"
               # scheduler = scheduler
               )
