import torch
import torch.nn as nn
import open_clip

class DualHeadCLIPViT(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes, clip_model_name="ViT-B-32-quickgelu", pretrained="openai"):
        """
        Custom model using CLIP (ViT-based) as frozen backbone and two classification heads.

        Args:
            num_species_classes (int): number of species classes
            num_disease_classes (int): number of disease classes
            clip_model_name (str): e.g., "ViT-B-32", "ViT-L-14", "ViT-bigG-14"
            pretrained (str): e.g., "openai", "laion2b_s32b_b79k", etc.
        """
        super(DualHeadCLIPViT, self).__init__()
        self.model_name = "CLIPViT"

        # Load pretrained CLIP with ViT backbone from open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )

        # Freeze all parameters of the CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get the dimensionality of the image embeddings
        embed_dim = self.clip_model.visual.output_dim  # e.g., 512 or 768 or 1024

        # Add two trainable classification heads
        self.species_head = nn.Linear(embed_dim, num_species_classes)
        self.disease_head = nn.Linear(embed_dim, num_disease_classes)

    def forward(self, x):
        """
        Forward pass: returns logits from frozen CLIP visual encoder + 2 heads

        Args:
            x (Tensor): input images [B, 3, 224, 224]

        Returns:
            Tuple[Tensor, Tensor]: species_logits, disease_logits
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # frozen CLIP backbone

        species_logits = self.species_head(features)
        disease_logits = self.disease_head(features)
        return species_logits, disease_logits

    def get_name(self):
        return self.model_name