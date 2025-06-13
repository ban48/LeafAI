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
    # model = DualHeadResNet(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(model)                                                       # DECOMMENT
    # trainer.training(2)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # -----------------------------------------------------------
    
    # STEP 2 - Obtain predictions (1st part) and use the LLM (2nd part)
    # -----------------------------------------------------------
    # # 1st
    model = DualHeadResNet(num_species_classes=12, num_disease_classes=20)                # DECOMMENT
    model.load_checkpoints()                                                            # DECOMMENT
    imgs, filenames, true_species, true_diseases = load_random_inference_image(model.get_name())
    
    correct_species = 0
    correct_disease = 0
    correct_both = 0
    correct_at_least_one = 0
    total = len(imgs)
    
    
    for img, true_s, true_d in zip(imgs, true_species, true_diseases):
        pred_s_idx, pred_d_idx = model.predict(img)
        pred_s, pred_d = get_label_names(pred_s_idx, pred_d_idx)

        is_s = pred_s == true_s
        is_d = pred_d == true_d

        correct_species += is_s
        correct_disease += is_d
        correct_both += is_s and is_d
        correct_at_least_one += is_s or is_d

    print(f"Percentuale specie corrette: {correct_species / total:.2%}")
    print(f"Percentuale malattie corrette: {correct_disease / total:.2%}")
    print(f"Percentuale almeno una corretta: {correct_at_least_one / total:.2%}")
    print(f"Percentuale entrambe corrette: {correct_both / total:.2%}")
    
    # # 2nd
    # giorgio = LeafConditionDescriber()                                                # DECOMMENT
    # messagefromgiorgio = giorgio.describe(species_pred, disease_pred)                 # DECOMMENT
    # print(messagefromgiorgio)                                                         # DECOMMENT
    # -----------------------------------------------------------

if __name__ == "__main__":
    main()