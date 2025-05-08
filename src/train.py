from torch.utils.data import DataLoader
from src.dataset import LeafDataset

# Paths to CSVs
train_csv = "data/train.csv"
val_csv = "data/val.csv"

# Instantiate datasets
train_dataset = LeafDataset(csv_path=train_csv, mode="train")
val_dataset = LeafDataset(csv_path=val_csv, mode="val")

# -----------------------------------------------
# Create DataLoaders

# The DataLoader is not a list or an array of all batches.
# Instead, it is an iterable object that yields ONE batch at a time.
# 
# When we loop over the DataLoader:
# it calls the Dataset's __getitem__() multiple times per batch,
# builds tensors, and returns a batch as a tuple of tensors:
#   - images:  [B, 3, 224, 224]
#   - species: [B]
#   - disease: [B]
#
# This is memory-efficient: only one batch is loaded into RAM at a time.
# Internally, the DataLoader keeps track of which batch we're on.
# We don't need to manage indices or track progress ourselves.
#
# -----------------------------------------------

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

def _testPreProcessing():
    for images, species_labels, disease_labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Species labels:", species_labels)
        print("Disease labels:", disease_labels)
        break