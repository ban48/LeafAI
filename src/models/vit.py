import torch
import torch.nn as nn
import torchvision.models as models

class DualHeadViT(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes):
        """
        A custom Vision Transformer (ViT-B/16) model with two output heads:
        one for species classification and one for disease classification.

        Args:
            num_species_classes (int): number of unique plant species
            num_disease_classes (int): number of unique diseases
        """
        super(DualHeadViT, self).__init__()
        self.model_name = "ViT"

        # Load pretrained ViT-B/16 from torchvision
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # Extract the number of features from the classification token (cls_token)
        num_features = self.backbone.heads.head.in_features

        # Remove the original classification head
        self.backbone.heads.head = nn.Identity()  # Outputs only the [CLS] embedding

        # Add two separate classification heads
        self.species_head = nn.Linear(num_features, num_species_classes)
        self.disease_head = nn.Linear(num_features, num_disease_classes)

    def forward(self, x):
        """
        Forward pass: returns a tuple of logits for both tasks.

        Args:
            x (Tensor): input images of shape [B, 3, 224, 224]

        Returns:
            Tuple[Tensor, Tensor]:
                - species_logits: [B, num_species_classes]
                - disease_logits: [B, num_disease_classes]
        """
        # Network fed with input batch
        features = self.backbone(x)

        # Final probabilities vectors for species and diseases 
        species_logits = self.species_head(features)
        disease_logits = self.disease_head(features)
        
        return species_logits, disease_logits
    
    def load_checkpoints(self, checkpoint_path=None, device="cpu"):
        if checkpoint_path is None:
            checkpoint_path = f"checkpoints/{self.model_name}/best_model.pt"
        # Carica pesi fine-tuned
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        self.to(device)
    
    def predict(self, image_tensor, device = "cpu"):
        """
        Esegue l'inferenza su un'immagine singola.

        Args:
            image_tensor (torch.Tensor): tensor di input [1, 3, 224, 224]
            checkpoint_path (str): path al checkpoint fine-tuned
            device (torch.device): CPU / CUDA / MPS

        Returns:
            Tuple[int, int]: predizione (class index specie, class index malattia)
        """
        # Prepara input
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            species_logits, disease_logits = self(image_tensor)
            species_pred = torch.argmax(species_logits, dim=1).item()
            disease_pred = torch.argmax(disease_logits, dim=1).item()

        return species_pred, disease_pred
    
    def get_name(self):
        return self.model_name