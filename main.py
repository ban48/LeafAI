# main.py

# Import function to generate CSVs from dataset folders
from src.utils import generate_split_csvs

def main():
    # STEP 0 - Generate CSV files from dataset (run once, then comment this block)
    # -----------------------------------------------------------
    generate_split_csvs(                                                              # DECOMMENT
        base_dir="data/raw/PlantVillage",                                             # DECOMMENT
        output_dir="data"                                                             # DECOMMENT
    )                                                                                 # DECOMMENT
    print("[INFO] CSVs generated. You can now comment this section.")                 # DECOMMENT
    # -----------------------------------------------------------

    # STEP 1 - Continue with training pipeline, dataset loading, etc.
    print("[INFO] Main script is running...")


if __name__ == "__main__":
    main()