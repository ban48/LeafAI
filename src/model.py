import torch
import torch.nn as nn
import torchvision.models as models

class DualHeadResNet(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes):
        """
        A custom ResNet18-based model with two output heads:
        one for species classification and one for disease classification.

        Args:
            num_species_classes (int): number of unique plant species
            num_disease_classes (int): number of unique diseases
        """
        super(DualHeadResNet, self).__init__()

        # Load pretrained ResNet18 from torchvision
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Extract the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Remove the original classification layer
        self.backbone.fc = nn.Identity()  # Outputs only the feature vector

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
        features = self.backbone(x)
        species_logits = self.species_head(features)
        disease_logits = self.disease_head(features)
        return species_logits, disease_logits