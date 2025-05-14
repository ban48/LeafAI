import torch
import torch.nn as nn
import open_clip

class DualHeadCLIPViT(nn.Module):
    def __init__(self, num_species_classes, num_disease_classes, clip_model_name="ViT-B-32-quickgelu", pretrained="openai"):
        """
        Custom model using CLIP (ViT-based) as frozen backbone and two classification heads. Only the image part is used.

        Args:
            num_species_classes (int): number of species classes
            num_disease_classes (int): number of disease classes
            clip_model_name (str): CLIP backbone variant (ViT in this case)
            pretrained (str): which pretrained weights to use
        """
        super(DualHeadCLIPViT, self).__init__()
        self.model_name = "CLIPViT"

        # Load pretrained CLIP with ViT backbone
        self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)

        # Freeze all parameters of the CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Get the dimensionality of the image embeddings, e.g. 512 or 768 or 1024
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