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
        self.model_name = "ResNet18"

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
        # Network fed with input batch
        features = self.backbone(x)

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
    
    
    
    
    
    
    