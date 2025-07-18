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
    for split in ["train", "val"]:
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

def load_inference_images(model_name: str, inference_subdir="data/raw/PlantVillage/inference"):
    """
    Load a random image from an inference directory and prepare it for inference

    Args:
        model_name (str): model used, e.g. "ResNet18", "CLIPViT", ...
        inference_subdir (str): relative directory for inference directory

    Returns:
        torch.Tensor: tensor [1, 3, 224, 224] (preprocessed image)
        str: file name
    """
    # Normalization map depending on model type
    stats = {
        "ResNet18":   ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "ViT":        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "DINOv2":     ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "CLIPResNet": ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
        "CLIPViT":    ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
        "default":    ([0.46986786, 0.49171314, 0.42178439], [0.17823954, 0.14965313, 0.19491802])
    }

    mean, std = stats.get(model_name, stats["default"])

    # Absolute path for inference directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inference_dir = os.path.join(base_dir, "..", inference_subdir)
    inference_dir = os.path.abspath(inference_dir)

    # Read avaible images
    image_files = [f for f in os.listdir(inference_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {inference_dir}")

    tensors, filenames, true_species, true_diseases = [], [], [], []

    for fname in image_files:
        img = cv2.imread(os.path.join(inference_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = (img - np.array(mean)) / np.array(std)
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float())
        filenames.append(fname)

        label_raw = fname.split("-")[0]  # remove eventual number
        species, disease = label_raw.split("___")
        true_species.append(species)
        true_diseases.append(disease)

    return tensors, filenames, true_species, true_diseases

def get_label_names(species_idx, disease_idx, csv_path="data/class_counts_summary.csv"):
    df = pd.read_csv(csv_path)

    # Sorted list of names
    species_list = sorted(df['species'].unique())
    disease_list = sorted(df['disease'].unique())

    species_name = species_list[species_idx]
    disease_name = disease_list[disease_idx]

    return species_name, disease_name


def evaluate_topk_accuracy(
    topk_species_preds: list[list[int]],
    topk_disease_preds: list[list[int]],
    true_species: list[str],
    true_diseases: list[str]
) -> dict:
    """
    Compute accuracy metrics using top-k predictions in index format,
    comparing names with get_label_names.
    """
    total = len(true_species)
    species_correct = 0
    disease_correct = 0
    at_least_one_correct = 0
    both_correct = 0

    for i in range(total):
        # Compare real name with the predicted ones
        pred_species_names = [get_label_names(idx, 0)[0] for idx in topk_species_preds[i]]
        pred_disease_names = [get_label_names(0, idx)[1] for idx in topk_disease_preds[i]]

        sp_correct = true_species[i] in pred_species_names
        ds_correct = true_diseases[i] in pred_disease_names

        species_correct += sp_correct
        disease_correct += ds_correct
        at_least_one_correct += sp_correct or ds_correct
        both_correct += sp_correct and ds_correct

    return {
        "species_topk_acc": species_correct / total,
        "disease_topk_acc": disease_correct / total,
        "at_least_one_topk_acc": at_least_one_correct / total,
        "both_correct_topk_acc": both_correct / total
    }