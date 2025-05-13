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
    
    def get_name(self):
        return self.model_name