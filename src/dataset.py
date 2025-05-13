import pandas as pd
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, dataset_type, csv_path, mode='train'):
        """
        Custom dataset that uses OpenCV for image loading and applies augmentations.

        Args:
            csv_path (str): Path to the CSV file (train or val).
            mode (str): 'train' or 'val'. Augmentations are only applied in 'train' mode.
        """
        self.df = pd.read_csv(csv_path)
        self.mode = mode

        # Label encoders
        self.species_list = sorted(self.df['species'].unique()) # Takes all the species and orders them alphabetically
        self.disease_list = sorted(self.df['disease'].unique()) # Takes all the disease and orders them alphabetically

        # Creates a dictionary mapping all the species to a integer and uniqe value
        self.species2idx = {}
        for idx, label in enumerate(self.species_list):
            self.species2idx[label] = idx                   

        # Creates a dictionary mapping all the disease to a integer and uniqe value
        self.disease2idx = {}
        for idx, label in enumerate(self.disease_list):
            self.disease2idx[label] = idx

        # Dataset-specific mean and std (R, G, B)
        if (dataset_type == "ResNet18" or dataset_type == "ViT" or dataset_type == "DINOv2"):
            self.mean=[0.485, 0.456, 0.406]
            self.std=[0.229, 0.224, 0.225]
        elif (dataset_type == "CLIPResNet" or dataset_type == "CLIPViT"):
            self.mean=[0.4815, 0.4578, 0.4082]
            self.std=[0.2686, 0.2613, 0.2758]
        else:
            self.mean=[0.46986786, 0.49171314, 0.42178439]
            self.std=[0.17823954, 0.14965313, 0.19491802]

    def __len__(self):
        return len(self.df)



    # --------------------------------------------------------------
    # __getitem__ is called automatically by the DataLoader.
    # 
    # It receives an index (idx), retrieves the corresponding row 
    # from the CSV, loads the image, applies preprocessing and 
    # augmentation (if in training mode), normalizes the image, 
    # converts it to a tensor, and returns a tuple:
    # 
    #   (image_tensor, species_index, disease_index)
    # --------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns a tuple: (normalized image tensor, species_label, disease_label)

        Args:
            idx (int): Index of the corresponding image row from the CSV, containing 3 columns (filepath, species, disease)
        """
        row = self.df.iloc[idx]
        img_path = row['filepath']

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # If in training mode, apply tranformations
        if self.mode == 'train':
            img = self.apply_augmentations(img)
            img = self.gaussian_blur(img)
        
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # Convert to torch tensor [C, H, W]
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

        species_idx = self.species2idx[row['species']]
        disease_idx = self.disease2idx[row['disease']]

        return img, species_idx, disease_idx

    def apply_augmentations(self, img):
        """
        Apply augmentations

        Args:
            img (MatLike): Image opened by opencv
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
    
    def gaussian_blur(self, img):
        return cv2.GaussianBlur(img, (3, 3), 0, cv2.BORDER_DEFAULT)