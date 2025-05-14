import os             
import pandas as pd   
import random
import cv2
import numpy as np
import torch  

def generate_split_csvs(base_dir="data/raw/PlantVillage/", output_dir="data"):
    """
    Generates two CSV files (train.csv and val.csv) 

    Each subfolder must contain class folders named like 'Species___Disease',
    each containing image files.

    Args:
        base_dir (str): Path to the root dataset folder containing 'train' and 'val'.
        output_dir (str): Path to the folder where the CSVs will be saved.
    """

    # Loop through each split (train and val)
    for split in ["/train", "/val"]:
        rows = []  # This list will store data for the current split

        # Path to the current split folder
        split_dir = os.path.join(base_dir, split)

        # Loop through all class folders (e.g., 'Apple___Black_rot')
        for class_dir in os.listdir(split_dir):

            # Skip folders that don't follow the expected naming convention
            if "___" not in class_dir:
                continue

            # Extract species and disease from folder name
            species, disease = class_dir.split("___")

            # Full path to the class folder
            class_path = os.path.join(split_dir, class_dir)

            # Loop through each image file inside the class folder
            for file_name in os.listdir(class_path):

                # Process only valid image files
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_path, file_name)

                    # Add a row with image path, species, and disease
                    rows.append({
                        "filepath": img_path,
                        "species": species,
                        "disease": disease
                    })

        # Convert the list of rows to a DataFrame
        df = pd.DataFrame(rows)

        # Make sure the output folder exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame as a CSV file
        output_csv = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(output_csv, index=False)

        # Print confirmation message
        print(f"[✓] Created {output_csv} with {len(df)} rows.")

def load_random_inference_image(model_name: str, inference_subdir="data/raw/PlantVillage/inference"):
    """
    Carica un'immagine random da una directory di inference e la prepara per l'inference.

    Args:
        model_name (str): modello usato, es. "ResNet18", "CLIPViT", etc.
        inference_subdir (str): percorso relativo alla cartella immagini (rispetto allo script chiamante)

    Returns:
        torch.Tensor: immagine preprocessata [1, 3, 224, 224]
        str: nome del file scelto
    """
    # Mappa delle normalizzazioni per tipo modello
    stats = {
        "ResNet18":   ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "ViT":        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "DINOv2":     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "CLIPResNet": ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
        "CLIPViT":    ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
        "default":    ([0.46986786, 0.49171314, 0.42178439], [0.17823954, 0.14965313, 0.19491802])
    }

    mean, std = stats.get(model_name, stats["default"])

    # Percorso assoluto alla cartella "inference"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inference_dir = os.path.join(base_dir, "..", inference_subdir)
    inference_dir = os.path.abspath(inference_dir)

    # Leggi immagini disponibili
    image_files = [f for f in os.listdir(inference_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {inference_dir}")

    # Scegli immagine random
    chosen_file = random.choice(image_files)
    full_path = os.path.join(inference_dir, chosen_file)

    # Carica immagine
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    img = img.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) # Converte in tensor [1, 3, 224, 224]

    return tensor, chosen_file

def get_label_names(species_idx, disease_idx, csv_path="data/class_counts_summary.csv"):
    df = pd.read_csv(csv_path)

    # Ricava le liste ordinate dei nomi
    species_list = sorted(df['species'].unique())
    disease_list = sorted(df['disease'].unique())

    species_name = species_list[species_idx]
    disease_name = disease_list[disease_idx]

    return species_name, disease_name