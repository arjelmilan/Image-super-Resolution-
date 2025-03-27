import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import dataset.common as common

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def save_model(model , optimizer , scheduler,results,epoch):
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'results': results,
  }
  torch.save(checkpoint, f'checkpoint{epoch}.pth')

def plot_train_loss_curves(results):

    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    epoch = results["epoch"]

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_loss, label="train_loss")
    plt.plot(epoch, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('loss.png')
    
def plot_psnr(results):
    avg_psnr = results["avg_psnr"]

    epochs = range(len(avg_psnr))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_psnr, label="avg_psnr")
    # plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('loss.png')
    

def plot_results(model,dataset):
  with torch.inference_mode():
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))

    for i in range(3):
        lr , hr = dataset[i]

        axes[0, i].imshow(common.tensor2np(lr))
        axes[0, i].axis('off')
        axes[0, i].set_title(f"LR Image {i+1}")

        # lr_tensor =
        sr = model(lr.unsqueeze(0).to(device))
        sr = common.tensor2np(sr.squeeze(0).detach().cpu())
        axes[1, i].imshow(sr)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"SR Image {i+1}")

        axes[2, i].imshow(common.tensor2np(hr))
        axes[2, i].axis('off')
        axes[2, i].set_title(f"Ground Truth {i+1}")

    plt.tight_layout()
    plt.show()

