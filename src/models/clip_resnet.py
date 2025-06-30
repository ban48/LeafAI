import torch
import torch.nn as nn
import open_clip

class DualHeadCLIPResNet(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes, clip_model_name="RN50-quickgelu", pretrained="openai"):
        """
        Custom model using CLIP (ResNet) as frozen backbone and two classification heads. Only the image part is used.

        Args:
            num_species_classes (int): number of species classes
            num_disease_classes (int): number of disease classes
            clip_model_name (str): CLIP backbone variant (ResNet50 in this case)
            pretrained (str): which pretrained weights to use
        """
        super(DualHeadCLIPResNet, self).__init__()
        self.model_name = "CLIPResNet"

        # Load pretrained CLIP with ResNet backbone
        self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)

        # Freeze all parameters of the CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get the dimensionality of the image embeddings, e.g. 1024 for RN50
        embed_dim = self.clip_model.visual.output_dim

        # Add two separate classification heads
        self.species_head = nn.Linear(embed_dim, num_species_classes)
        self.disease_head = nn.Linear(embed_dim, num_disease_classes)

    def forward(self, x):
        """
        Forward pass: returns logits from frozen CLIP visual encoder + 2 heads

        Args:
            x (Tensor): input images [B, 3, 224, 224]

        Returns:
            Tuple[Tensor, Tensor]:
                - species_logits: [B, num_species_classes]
                - disease_logits: [B, num_disease_classes]
        """

        # Frozen CLIP backbone (gradient not computed)
        with torch.no_grad():
            features = self.clip_model.encode_image(x) # CLS token automatically managed

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