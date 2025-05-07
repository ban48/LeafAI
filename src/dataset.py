import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, csv_path, mode='train'):
        """
        Custom dataset that uses OpenCV for image loading and applies augmentations
        consistent with the visualization logic used in the notebook.

        Args:
            csv_path (str): Path to the CSV file (train or val).
            mode (str): 'train' or 'val'. Augmentations are only applied in 'train' mode.
        """
        self.df = pd.read_csv(csv_path)
        self.mode = mode

        # Label encoders
        self.species_list = sorted(self.df['species'].unique())
        self.disease_list = sorted(self.df['disease'].unique())
        self.species2idx = {label: idx for idx, label in enumerate(self.species_list)}
        self.disease2idx = {label: idx for idx, label in enumerate(self.disease_list)}

        # Dataset-specific mean and std (R, G, B)
        self.mean = np.array([0.47131167, 0.49435022, 0.42405355])
        self.std  = np.array([0.17719288, 0.14827244, 0.19360321])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        print("SONO ENTRATOOOOOOOOOOO")
        """
        Returns a tuple: (normalized image tensor, species_label, disease_label)
        """
        row = self.df.iloc[idx]
        img_path = row['filepath']

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        if self.mode == 'train':
            img = self.apply_augmentations(img)

        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # Convert to torch tensor [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1)

        species_idx = self.species2idx[row['species']]
        disease_idx = self.disease2idx[row['disease']]

        return img, species_idx, disease_idx

    def apply_augmentations(self, img):
        """
        Apply augmentations
        """
        # Horizontal flip
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Saturation & Brightness (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        brightness_factor = np.random.uniform(0.7, 1.3)
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        saturation_factor = np.random.uniform(0.7, 1.3)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Contrast jitter (on RGB)
        contrast_factor = np.random.uniform(0.7, 1.3)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        return img