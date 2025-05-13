import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

class DualHeadDINOv2(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes, model_name="facebook/dinov2-base"):
        """
        Custom model using DINOv2 as frozen feature extractor with two classification heads.
        """
        super(DualHeadDINOv2, self).__init__()
        self.model_name = "DINOv2"

        # Load pre-trained DINOv2 model and image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = Dinov2Model.from_pretrained(model_name)

        # Freeze all DINOv2 parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the embedding dimension of the [CLS] token
        hidden_size = self.backbone.config.hidden_size

        # Two separate classification heads
        self.species_head = nn.Linear(hidden_size, num_species_classes)
        self.disease_head = nn.Linear(hidden_size, num_disease_classes)

    def forward(self, x):
        """
        Forward pass through DINOv2 and classification heads.
        Args:
            x (Tensor): input images [B, 3, H, W]

        Returns:
            Tuple[Tensor, Tensor]: species_logits, disease_logits
        """
        with torch.no_grad():
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

        species_logits = self.species_head(features)
        disease_logits = self.disease_head(features)
        return species_logits, disease_logits

    def get_name(self):
        return self.model_name