import torch
import torch.amp
from tqdm.auto import tqdm
import skimage.color as sc
import dataset.common as common



device = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALE = 2


def train_step(model:torch.nn.Module,
               dataloader : torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               scaler : torch.cuda.amp.GradScaler,
               scheduler: torch.optim.lr_scheduler,
               device):
  model.train()
  train_loss = 0
  for batch,(lr,hr) in enumerate(dataloader):
    X,y = lr.to(device),hr.to(device)
    # X = X.type(torch.float32)
    # y = y.type(torch.float32)

    with torch.amp.autocast(device):
      #1. Forward pass
      y_pred = model(X)

      #2. Calculate loss
      loss =loss_fn(y_pred,y)
      train_loss += loss.item()

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward
      # loss.backward()
    scaler.scale(loss).backward()

    #5. Update optimizer parameters
    scaler.step(optimizer)
    scaler.update()
    # optimizer.step()

    #5. Step the learning rate scheduler at the end of each epoch
  scheduler.step()

  train_loss /=len(dataloader)

  return train_loss

def valid(model:torch.nn.Module,
          valid_dataloader : torch.utils.data.DataLoader,
          loss_fn : torch.nn.Module,
          device):
    model.eval()

    valid_loss = 0
    with torch.inference_mode():
      for batch,(lr,hr) in enumerate(valid_dataloader):
        X,y = lr.to(device),hr.to(device)

        with torch.amp.autocast(device):
          y_pred = model(X)
          loss =loss_fn(y_pred,y)
          valid_loss += loss.item()

    valid_loss = valid_loss / len(valid_dataloader)

    return valid_loss

def test(model:torch.nn.Module,
          test_dataloader : torch.utils.data.DataLoader,
          device):
    model.eval()

    avg_psnr, avg_ssim  = 0, 0
    for batch in test_dataloader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if device == 'cuda':
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.inference_mode():
          with torch.amp.autocast(device):
            pre = model(lr_tensor)



        sr_img = common.tensor2np(pre.detach()[0])
        gt_img = common.tensor2np(hr_tensor.detach()[0])
        crop_size = SCALE
        cropped_sr_img = common.shave(sr_img, crop_size)
        cropped_gt_img = common.shave(gt_img, crop_size)
        im_label = common.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = common.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])

        avg_psnr += common.compute_psnr(im_pre, im_label)
        avg_ssim += common.compute_ssim(im_pre, im_label)

    avg_psnr = avg_psnr / len(test_dataloader)
    avg_ssim = avg_ssim / len(test_dataloader)

    return avg_psnr , avg_ssim




def train(model : torch.nn.Module,
          train_dataloader : torch.utils.data.DataLoader,
          valid_dataloader : torch.utils.data.DataLoader,
          test_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          loss_fn : torch.nn.Module,
          epochs : int,
          scaler : torch.cuda.amp.GradScaler,
          scheduler: torch.optim.lr_scheduler,
          device):
  results = {'epoch':[],
              'train_loss':[],
             'test_loss':[],
             'avg_psnr':[],
             'avg_ssim':[]}

  for epoch in tqdm(range(epochs)):
    train_loss = train_step(model= model,
                            dataloader=train_dataloader,
                            loss_fn = loss_fn,
                            optimizer = optimizer,
                            scaler = scaler,
                            scheduler = scheduler,
                            device = device)
    test_loss = valid(model= model,
                            valid_dataloader=valid_dataloader,
                            loss_fn = loss_fn,
                            device = device)
    avg_psnr , avg_ssim = test(model= model,
                            test_dataloader=test_dataloader,
                            device = device)


    if epoch % 1 == 0:
      print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"avg_psnr: {avg_psnr:.4f} | "
        f"avg_ssim: {avg_ssim:.4f}"
        )

      # Update results dictionary
    results["epoch"].append(epoch+1)
    results["train_loss"].append(train_loss)
    results["test_loss"].append(test_loss)
    results["avg_psnr"].append(avg_psnr)
    results["avg_ssim"].append(avg_ssim)


  return results
