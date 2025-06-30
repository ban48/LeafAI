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
    # print("[INFO] Training ResNet18")
    # resnet = DualHeadResNet(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(resnet)                                                       # DECOMMENT
    # trainer.training(30)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # print("[INFO] End training ResNet18")

    # print("[INFO] Training ViT")
    # vit = DualHeadViT(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(vit)                                                       # DECOMMENT
    # trainer.training(30)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # print("[INFO] End training ViT")

    # print("[INFO] Training CLIPResNet")
    # clipresnet = DualHeadCLIPResNet(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(clipresnet)                                                       # DECOMMENT
    # trainer.training(30)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # print("[INFO] End training CLIPResNet")

    # print("[INFO] Training CLIPViT")
    # clipvit = DualHeadCLIPViT(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(clipvit)                                                       # DECOMMENT
    # trainer.training(30)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # print("[INFO] End training CLIPViT")

    # print("[INFO] Training DINOv2")
    # dino = DualHeadDINOv2(num_species_classes=12, num_disease_classes=20)            # DECOMMENT
    # trainer = tr.Trainer(dino)                                                       # DECOMMENT
    # trainer.training(30)                                                               # DECOMMENT
    # trainer.list_all_checkpoints()                                                    # DECOMMENT
    # print("[INFO] End training DINOv2")
    # -----------------------------------------------------------
    
    # STEP 2 - Obtain predictions (1st part) and use the LLM (2nd part)
    # -----------------------------------------------------------
    # # 1st
    # Choose the trained model
    model = DualHeadDINOv2(num_species_classes=12, num_disease_classes=20)                # DECOMMENT
    model.load_checkpoints()                                                            # DECOMMENT
    imgs, filenames, true_species, true_diseases = load_inference_images(model.get_name())
    
    # Where top-1 and top-3 will be stored
    top1_species_preds, top3_species_preds = [], []
    top1_disease_preds, top3_disease_preds = [], []

    # Predictions
    for img in imgs:
        species_topk, disease_topk = model.predict_topk(img, k=3)

        # Saving top-1 (first element) and top-3
        top1_species_preds.append([species_topk[0]])
        top3_species_preds.append(species_topk)

        top1_disease_preds.append([disease_topk[0]])
        top3_disease_preds.append(disease_topk)

    # Evaluation
    print("\n")
    print(model.get_name())
    print("[Correct Class Accuracy]")
    results_top1 = evaluate_topk_accuracy(top1_species_preds, top1_disease_preds, true_species, true_diseases)
    for key, value in results_top1.items():
        print(f"{key}: {value:.2%}")

    print("\n[Top-3 Accuracy]")
    results_top3 = evaluate_topk_accuracy(top3_species_preds, top3_disease_preds, true_species, true_diseases)
    for key, value in results_top3.items():
        print(f"{key}: {value:.2%}")
    
    # # 2nd
    # Use tinyllama for natural language inference 
    # giorgio = LeafConditionDescriber()                                                # DECOMMENT
    # messagefromgiorgio = giorgio.describe(species_pred, disease_pred)                 # DECOMMENT
    # print(messagefromgiorgio)                                                         # DECOMMENT
    # -----------------------------------------------------------
    
    print("[INFO] Main script finished")

if __name__ == "__main__":
    main()