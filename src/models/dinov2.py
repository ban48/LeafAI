import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model

class DualHeadDINOv2(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes, model_name="facebook/dinov2-base"):
        """
        Custom model using DINOv2 as frozen feature extractor with two classification heads.

        Args:
            num_species_classes (int): number of species classes
            num_disease_classes (int): number of disease classes
            model_name (str): DINOv2 variant
        """
        super(DualHeadDINOv2, self).__init__()
        self.model_name = "DINOv2"

        # Load pre-trained DINOv2 model and image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = Dinov2Model.from_pretrained(model_name)

        # Freeze all the parameters of DINOv2
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the dimensionality of the image embeddings
        hidden_size = self.backbone.config.hidden_size

        # Add two separate classification heads
        self.species_head = nn.Linear(hidden_size, num_species_classes)
        self.disease_head = nn.Linear(hidden_size, num_disease_classes)

    def forward(self, x):
        """
        Forward pass through frozen DINOv2 and classification heads.

        Args:
            x (Tensor): input images [B, 3, 224, 224]

        Returns:
            Tuple[Tensor, Tensor]:
                - species_logits: [B, num_species_classes]
                - disease_logits: [B, num_disease_classes]
        """

        # Frozen backbone (gradient not computed)
        with torch.no_grad():
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token for classification - not automatically managed

        # Final probabilities vectors for species and diseases
        species_logits = self.species_head(features)
        disease_logits = self.disease_head(features)
        return species_logits, disease_logits
    
    def load_checkpoints(self, checkpoint_path=None, device="cpu"):
        if checkpoint_path is None:
            checkpoint_path = f"checkpoints/{self.model_name}/best_model.pt"

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        self.to(device)
    
    def predict_topk(self, image_tensor,  k=1, device = "cpu"):
        """
        Inference on single image, obtaining the top-k accuracy

        Args:
            image_tensor (torch.Tensor): input tensor [1, 3, 224, 224]
            k (int): how many best accuracies to pick
            device (torch.device): CPU / CUDA / MPS

        Returns:
            Tuple[List[int], List[int]]: top-k predicted classes for species e disease
        """
        # Prepare input
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            species_logits, disease_logits = self(image_tensor)
            topk_species = torch.topk(species_logits, k, dim=1).indices.squeeze(0).tolist()
            topk_diseases = torch.topk(disease_logits, k, dim=1).indices.squeeze(0).tolist()

        return topk_species, topk_diseases

    def get_name(self):
        return self.model_name