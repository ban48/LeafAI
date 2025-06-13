import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import LeafDataset
from sklearn.metrics import f1_score
import glob



class Trainer:
    def __init__(self, model, train_csv="data/train.csv", val_csv="data/val.csv", batch_size=32, lr=1e-3):
        """
        Training class specifically for our ResNet-18

        Args:
            model (Module): model we want to train (ResNet, ViT, CLIP or DINOv2)
            train_csv (str): Path training CSV file
            val_csv (str): Path validation CSV file
            batch_size (int): Size of input batch (number of input images at time)
            lr (float): Learning rate for parameters optimization
        """

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
        # Init
        # ---------------------------------------------------------------
        self.model = model
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.lr = lr
        self.patience = 5               # For early stopping
        self.early_stop_counter = 0
        
        # Get model name from class and create specific checkpoint dir
        self.model_name = model.get_name()
        self.checkpoint_dir = f"checkpoints/{self.model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Path to resume the best parameters for the current model
        self.param_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        self._setup()



    def _setup(self):
        # ---------------------------------------------------------------
        # Instantiate datasets
        # ---------------------------------------------------------------
        train_dataset = LeafDataset(self.model_name, csv_path=self.train_csv, mode="train")
        val_dataset   = LeafDataset(self.model_name, csv_path=self.val_csv, mode="val")


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
        self.model.to(self.device)

        self.loss_species_calc = nn.CrossEntropyLoss() 
        self.loss_diseases_calc = nn.CrossEntropyLoss() 

        # Used Adam as GD
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.start_epoch = 0 
        
        self.best_f1_species = 0.0
        self.best_f1_disease = 0.0
        self.best_f1_macro = 0.0

        self.best_acc_species = 0.0
        self.best_acc_disease = 0.0
        self.best_acc_avg = 0.0
        
        self.current_loss_species = 0.0
        self.current_loss_disease = 0.0
        self.current_loss = 0.0
        
        # Resume training if checkpoint exists
        if os.path.exists(self.param_path):
            checkpoint = torch.load(self.param_path, map_location=self.device)                   # Loads the checkpoint file and maps the tensors to the current device (CPU, CUDA, MPS)
            self.model.load_state_dict(checkpoint["model_state_dict"])                           # Restores the model weights from the checkpoint
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])                   # Restores the optimizer state (e.g. learning rate, momentum, etc.) so training resumes correctly
            self.start_epoch = checkpoint["epoch"] + 1                                           # Sets the starting epoch to resume from (we add +1 because epochs are zero-indexed)
            
            self.best_f1_species = checkpoint["best_f1_species"]
            self.best_f1_disease = checkpoint["best_f1_disease"]
            self.best_f1_macro = checkpoint["best_f1_macro"]

            self.best_acc_species = checkpoint["best_acc_species"]
            self.best_acc_disease = checkpoint["best_acc_disease"]
            self.best_acc_avg = checkpoint["best_acc_avg"]
            print(f"[INFO] Resumed from checkpoint: epoch {self.start_epoch} (best acc: species={self.best_acc_species:.4f}, disease={self.best_acc_disease:.4f}, avg={self.best_acc_avg:.4f})")

        
        
    def training(self, num_epochs=10):
        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):

            # Prepare the model for training (e.g. enables dropout, batchnorm ...)
            self.model.train()     
            total_loss = 0.0
            species_loss_sum = 0.0
            disease_loss_sum = 0.0
            num_batches = len(self.train_loader)              

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
                self.optimizer.zero_grad()      # Resets the gradient for each epoch
                loss.backward()                 # Computes the gradient
                self.optimizer.step()           # Updates the weights of the model

                total_loss += loss.item()
                species_loss_sum += loss_species.item()
                disease_loss_sum += loss_disease.item()
            
            
            self.current_loss = total_loss / num_batches
            self.current_loss_species = species_loss_sum / num_batches
            self.current_loss_disease = disease_loss_sum / num_batches

            print(f"[Epoch {epoch+1}] Train Loss: {self.current_loss:.4f} (species: {self.current_loss_species:.4f}, disease: {self.current_loss_disease:.4f})")
            
            self.validate(epoch)



    def validate(self, epoch):
        # -----------------------------
        # Validation
        # -----------------------------

        # Prepare the model for val (e.g. disables dropout, batchnorm ...)
        self.model.eval()                    
        correct_species = 0
        correct_disease = 0
        total = 0
        
        all_true_species = []
        all_pred_species = []
        all_true_disease = []
        all_pred_disease = []

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
                
                all_true_species.extend(species_labels.cpu().numpy())
                all_pred_species.extend(pred_species.cpu().numpy())
                all_true_disease.extend(disease_labels.cpu().numpy())
                all_pred_disease.extend(pred_disease.cpu().numpy())

        acc_species = correct_species / total
        acc_disease = correct_disease / total
        avg_val_acc = (acc_species + acc_disease) / 2
        
        f1_species = f1_score(all_true_species, all_pred_species, average='macro')
        f1_disease = f1_score(all_true_disease, all_pred_disease, average='macro')
        f1_macro = (f1_species + f1_disease) / 2

        print(f"[Epoch {epoch+1}] Val Accuracy - Species: {acc_species:.4f} | Disease: {acc_disease:.4f}")
        print(f"[Epoch {epoch+1}] Val F1 Macro  - Species: {f1_species:.4f} | Disease: {f1_disease:.4f} | Avg: {f1_macro:.4f}")

        # Early stopping logic
        if f1_macro <= self.best_f1_macro:
            self.early_stop_counter += 1
            print(f"[INFO] No improvement. Early stop counter: {self.early_stop_counter}/{self.patience}")
            if self.early_stop_counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                exit(0)
        else:
            self.early_stop_counter = 0


        # ---------------------------------------------
        # Save best model checkpoint
        # ---------------------------------------------

        # Save model parameters only if it's the best so far
        if f1_macro > self.best_f1_macro:
            self.best_f1_macro = f1_macro
            self.best_f1_species = f1_species
            self.best_f1_disease = f1_disease
            self.best_acc_species = acc_species
            self.best_acc_disease = acc_disease
            self.best_acc_avg = avg_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),

                # F1 Macro 
                "best_f1_species": f1_species,
                "best_f1_disease": f1_disease,
                "best_f1_macro": f1_macro,

                # Accuracy 
                "best_acc_species": acc_species,
                "best_acc_disease": acc_disease,
                "best_acc_avg": avg_val_acc,
            }, self.param_path)
            print(f"[INFO] New best model saved to {self.param_path}")
        
        # Save training log of current epoch (no model weights)
        log_path = os.path.join(self.checkpoint_dir, f"log_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            
            # Current F1 scores
            "current_f1_species": f1_species,
            "current_f1_disease": f1_disease,
            "current_f1_macro": f1_macro,

            # Current Accuracies
            "current_acc_species": acc_species,
            "current_acc_disease": acc_disease,
            "current_acc_avg": avg_val_acc,
            
            # Current Loss
            "current_loss_species": self.current_loss_species,
            "current_loss_disease": self.current_loss_disease,
            "current_loss": self.current_loss,

            # Best F1
            "best_f1_species": self.best_f1_species,
            "best_f1_disease": self.best_f1_disease,
            "best_f1_macro": self.best_f1_macro,

            # Best Accuracy
            "best_acc_species": self.best_acc_species,
            "best_acc_disease": self.best_acc_disease,
            "best_acc_avg": self.best_acc_avg,
            
        }, log_path)
        print(f"[INFO] Checkpoint saved: {epoch} acc_species: {acc_species} acc_disease: {acc_disease} acc_average: {avg_val_acc}")
        print(f"[INFO] Checkpoint saved: {epoch} f1_species: {f1_species} f1_disease: {f1_disease} f1_average: {f1_macro}")
            
        
                  
    def list_all_checkpoints(self):
        """
        Show all my checkpoints for the current model
        """
        checkpoint_files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "log_epoch_*.pt")))
        print("\n[INFO] Saved checkpoints:")
        
        for file in checkpoint_files:
            checkpoint = torch.load(file, map_location="cpu")
            print(f" - {file}:")
            print(f"     Epoch:                    {checkpoint['epoch']+1}")
            print(f"     Val Accuracy (Species):   {checkpoint['current_acc_species']:.4f}")
            print(f"     Val Accuracy (Disease):   {checkpoint['current_acc_disease']:.4f}")
            print(f"     Val Accuracy (Average):   {checkpoint['current_acc_avg']:.4f}")
            print(f"     Val F1 Macro (Species):   {checkpoint['current_f1_species']:.4f}")
            print(f"     Val F1 Macro (Disease):   {checkpoint['current_f1_disease']:.4f}")
            print(f"     Val F1 Macro (Average):   {checkpoint['current_f1_macro']:.4f}")
            print(f"     Best Accuracy (Species):  {checkpoint['best_acc_species']:.4f}")
            print(f"     Best Accuracy (Disease):  {checkpoint['best_acc_disease']:.4f}")
            print(f"     Best Accuracy (Average):  {checkpoint['best_acc_avg']:.4f}")
            print(f"     Best F1 Macro (Species):  {checkpoint['best_f1_species']:.4f}")
            print(f"     Best F1 Macro (Disease):  {checkpoint['best_f1_disease']:.4f}")
            print(f"     Best F1 Macro (Average):  {checkpoint['best_f1_macro']:.4f}")
            print(f"     Loss (Species):  {checkpoint['current_loss_species']:.4f}")
            print(f"     Loss (Disease):  {checkpoint['current_loss_disease']:.4f}")
            print(f"     Loss (Average):  {checkpoint['current_loss']:.4f}")