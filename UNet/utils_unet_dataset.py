import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class UNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = [
            os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png"))
            for p in self.img_paths
        ]
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.transform(img)    # Tensor, shape: [1, H, W], range: [0,1]
        mask = self.transform(mask)  # Tensor, shape: [1, H, W], but may contain [0,255]

        # ✅ 마스크를 0/1 binary로 변환
        mask = (mask > 0).float()

        return img, mask
