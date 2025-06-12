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
    
    print("[INFO] Main script is running...")
    
    # STEP 0 - Generate CSV files from dataset (run once, then comment this block)
    # -----------------------------------------------------------
    # generate_split_csvs(                                                              # DECOMMENT
    #     base_dir="data/raw/PlantVillage",                                             # DECOMMENT
    #     output_dir="data"                                                             # DECOMMENT
    # )                                                                                 # DECOMMENT
    # print("[INFO] CSVs generated. You can now comment this section.")                 # DECOMMENT
    # -----------------------------------------------------------
    
    # STEP 1 - Train the network
    # -----------------------------------------------------------
    model = DualHeadResNet(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    trainer = tr.Trainer(model)                                                       # DECOMMENT
    trainer.training(2)                                                               # DECOMMENT
    trainer.list_all_checkpoints()                                                    # DECOMMENT
    # -----------------------------------------------------------
    
    # STEP 2 - Obtain predictions (1st part) and use the LLM (2nd part)
    # -----------------------------------------------------------
    # # 1st
    # model = DualHeadResNet(num_species_classes=12, num_disease_classes=20)                # DECOMMENT
    # model.load_checkpoints()                                                            # DECOMMENT
    # imgs, filenames =  load_random_inference_image(model.get_name()) 
    # for img,filename in zip(imgs, filenames):                                                                   # DECOMMENT
    #     species_pred_idx, disease_pred_idx = model.predict(img)                           # DECOMMENT
    #     species_pred, disease_pred = get_label_names(species_pred_idx, disease_pred_idx)  # DECOMMENT
    #     print(filename)                                                                   # DECOMMENT
    #     print(species_pred)                                                               # DECOMMENT
    #     print(disease_pred)                                                               # DECOMMENT
    #     print("\n")                                                                       # DECOMMENT
    
    # # 2nd
    # giorgio = LeafConditionDescriber()                                                # DECOMMENT
    # messagefromgiorgio = giorgio.describe(species_pred, disease_pred)                 # DECOMMENT
    # print(messagefromgiorgio)                                                         # DECOMMENT
    # -----------------------------------------------------------

if __name__ == "__main__":
    main()