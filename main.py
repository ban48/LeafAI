from src.utils import generate_split_csvs
from src.utils import load_inference_images
from src.utils import evaluate_topk_accuracy
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
    imgs, filenames, true_species, true_diseases = load_inference_images(model.get_name())
    
    # Contenitori per top-1 e top-3
    top1_species_preds, top3_species_preds = [], []
    top1_disease_preds, top3_disease_preds = [], []

    # Predizioni
    for img in imgs:
        species_topk, disease_topk = model.predict_topk(img, k=3)

        # Salvo top-1 (primo elemento) e top-3
        top1_species_preds.append([species_topk[0]])
        top3_species_preds.append(species_topk)

        top1_disease_preds.append([disease_topk[0]])
        top3_disease_preds.append(disease_topk)

    # Valutazioni
    print("\n[Correct Class Accuracy]")
    results_top1 = evaluate_topk_accuracy(top1_species_preds, top1_disease_preds, true_species, true_diseases)
    for key, value in results_top1.items():
        print(f"{key}: {value:.2%}")

    print("\n[Top-3 Accuracy]")
    results_top3 = evaluate_topk_accuracy(top3_species_preds, top3_disease_preds, true_species, true_diseases)
    for key, value in results_top3.items():
        print(f"{key}: {value:.2%}")
    
    # # 2nd
    # giorgio = LeafConditionDescriber()                                                # DECOMMENT
    # messagefromgiorgio = giorgio.describe(species_pred, disease_pred)                 # DECOMMENT
    # print(messagefromgiorgio)                                                         # DECOMMENT
    # -----------------------------------------------------------

if __name__ == "__main__":
    main()