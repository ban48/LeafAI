import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import DualHeadResNet
from src.dataset import LeafDataset


class ResnetTrainer:
    def __init__(self, train_csv="data/train.csv", val_csv="data/val.csv", batch_size=32, lr=1e-3):
        # ---------------------------------------------------------------
        # Device selection – supports MPS (Mac), CUDA (NVIDIA), or CPU
        # ---------------------------------------------------------------
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[INFO] Using Apple MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[INFO] Using NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            print("[INFO] Using CPU (no GPU detected)")

        # ---------------------------------------------------------------
        # Paths to CSVs
        # ---------------------------------------------------------------
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.lr = lr
        
        self._setup()

    def _setup(self):
        # ---------------------------------------------------------------
        # Instantiate datasets
        # ---------------------------------------------------------------
        train_dataset = LeafDataset(csv_path=self.train_csv, mode="train")
        val_dataset   = LeafDataset(csv_path=self.val_csv, mode="val")

        # ---------------------------------------------------------------
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
        # ---------------------------------------------------------------

        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=4)
        self.val_loader   = DataLoader(val_dataset, self.batch_size, shuffle=False, num_workers=4)

        # ---------------------------------------------------------------
        # Model, Loss and Optimizer Setup
        # ---------------------------------------------------------------

        num_species = len(train_dataset.species2idx)
        num_disease = len(train_dataset.disease2idx)

        self.model = DualHeadResNet(num_species_classes=num_species, num_disease_classes=num_disease)
        self.model.to(self.device)

        self.loss_species_calc = nn.CrossEntropyLoss() 
        self.loss_diseases_calc = nn.CrossEntropyLoss() 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Directory to save best model
        os.makedirs("checkpoints", exist_ok=True)
        self.best_val_acc = 0.0
        
    def training(self, num_epochs=10):
        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------
        for epoch in range(num_epochs):
            self.model.train()                   # Prepare the model for training (e.g. enables dropout, batchnorm ...)
            running_loss = 0.0

            for images, species_labels, disease_labels in self.train_loader:
                
                # Move tensors on the selected device
                images = images.to(self.device)                          
                species_labels = species_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)

                # Forward pass - we obtain the unormalized probability vector (softmax is inside the entropy function)
                species_logits, disease_logits = self.model(images) 

                # Compute loss
                loss_species = self.loss_species_calc(species_logits, species_labels) 
                loss_disease = self.loss_diseases_calc(disease_logits, disease_labels) 
                loss = loss_species + loss_disease

                # Backward and optimize
                self.optimizer.zero_grad()       # Resets the gradient for each epoch
                loss.backward()             # Computes the gradient
                self.optimizer.step()            # Updates the weights of the model

                running_loss += loss.item()
                

            avg_train_loss = running_loss / len(self.train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
            self.validate(epoch)


    def validate(self, epoch):
            # --------------------------
            # Validation loop
            # --------------------------
            self.model.eval()                    # Prepare the model for val (e.g. disables dropout, batchnorm ...)
            correct_species = 0
            correct_disease = 0
            total = 0

            with torch.no_grad():           # Disables gradient tracking 
                for images, species_labels, disease_labels in self.val_loader:
                    images = images.to(self.device)
                    species_labels = species_labels.to(self.device)
                    disease_labels = disease_labels.to(self.device)

                    species_logits, disease_logits = self.model(images)

                    # Get predictions - Takes the index of the highest logit (predicted class)
                    _, pred_species = torch.max(species_logits, 1)      
                    _, pred_disease = torch.max(disease_logits, 1)

                    correct_species += (pred_species == species_labels).sum().item()
                    correct_disease += (pred_disease == disease_labels).sum().item()
                    total += species_labels.size(0)

            acc_species = correct_species / total
            acc_disease = correct_disease / total
            avg_val_acc = (acc_species + acc_disease) / 2

            print(f"[Epoch {epoch+1}] Val Accuracy - Species: {acc_species:.4f} | Disease: {acc_disease:.4f}")

            # --------------------------
            # Save best model checkpoint
            # --------------------------
            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc
                torch.save(self.model.state_dict(), "checkpoints/best_model.pt")
                print(f"Best model saved with accuracy: {self.best_val_acc:.4f}")