import os
import random
from PIL import Image

def create_patches(input_dir, lr_dir, hr_dir, patch_size, num_patches):
    """
    Creates random patches of LR and HR images and saves them in separate directories.

    Parameters:
    - input_dir (str): Directory containing the original high-resolution images.
    - lr_dir (str): Directory to save the low-resolution patches.
    - hr_dir (str): Directory to save the high-resolution patches.
    - patch_size (int): Size of the patches (e.g., 64 for 64x64).
    - num_patches (int): Number of patches to create per image.
    """
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Ensure image is in RGB format
                width, height = img.size

                for i in range(num_patches):
                    # Randomly select top-left corner for the patch
                    x = random.randint(0, width - patch_size)
                    y = random.randint(0, height - patch_size)

                    # Extract high-resolution patch
                    hr_patch = img.crop((x, y, x + patch_size, y + patch_size))

                    # Downscale to low-resolution patch
                    lr_patch = hr_patch.imresize((patch_size // 2, patch_size // 2), Image.BICUBIC)

                    # Save patches
                    hr_patch.save(os.path.join(hr_dir, f"{os.path.splitext(img_name)[0]}_hr_{i}.png"))
                    lr_patch.save(os.path.join(lr_dir, f"{os.path.splitext(img_name)[0]}_lr_{i}.png"))

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Example usage
input_dir = "./images"  # Directory with original HR images
lr_dir = "./lr_patches"  # Directory to save LR patches
hr_dir = "./hr_patches"  # Directory to save HR patches
patch_size = 64  # Patch size (64x64)
num_patches = 100  # Number of patches per image

create_patches(input_dir, lr_dir, hr_dir, patch_size, num_patches)
