from torch.utils.data import DataLoader
from src.dataset import LeafDataset

# Paths to CSVs
train_csv = "data/train.csv"
val_csv = "data/val.csv"

# Instantiate datasets
train_dataset = LeafDataset(csv_path=train_csv, mode="train")
val_dataset = LeafDataset(csv_path=val_csv, mode="val")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def _testPreProcessing():
    for images, species_labels, disease_labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Species labels:", species_labels)
        print("Disease labels:", disease_labels)
        break