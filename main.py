# main.py

# Import function to generate CSVs from dataset folders
from src.utils import generate_split_csvs
from src.utils import load_random_inference_image
from src.utils import get_label_names

from src.models.dinov2 import DualHeadDINOv2
from src.models.clip_vit import DualHeadCLIPViT
from src.models.clip_resnet import DualHeadCLIPResNet
from src.models.resnet18 import DualHeadResNet
from src.models.vit import DualHeadViT

from src.llm_inference import LeafConditionDescriber

from src.dataset import LeafDataset

import src.train as tr

def main():
    # STEP 0 - Generate CSV files from dataset (run once, then comment this block)
    # -----------------------------------------------------------
    # generate_split_csvs(                                                              # DECOMMENT
    #     base_dir="data/raw/PlantVillage",                                             # DECOMMENT
    #     output_dir="data"                                                             # DECOMMENT
    # )                                                                                 # DECOMMENT
    # print("[INFO] CSVs generated. You can now comment this section.")                 # DECOMMENT
    # -----------------------------------------------------------

    # STEP 1 - Continue with training pipeline, dataset loading, etc.
    print("[INFO] Main script is running...")
    
    model = DualHeadResNet(num_species_classes=12, num_disease_classes=20)
    model.load_checkpoints()
    
    img, filename =  load_random_inference_image(model.get_name())
    species_pred_idx, disease_pred_idx = model.predict(img)
    species_pred, disease_pred = get_label_names(species_pred_idx, disease_pred_idx)
    print(filename)
    print(species_pred)
    print(disease_pred)
    
    
    # trainer = tr.Trainer(model)
    # trainer.training(1)
    # trainer.list_all_checkpoints()
    
    # giorgio = LeafConditionDescriber()
    # messagefromgiorgio = giorgio.describe(species_pred, disease_pred)
    # print(messagefromgiorgio)
    


if __name__ == "__main__":
    main()