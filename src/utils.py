import os             
import pandas as pd     

def generate_split_csvs(base_dir="data/raw/PlantVillage", output_dir="data"):
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