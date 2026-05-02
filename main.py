import os
import argparse
import pandas as pd
from tqdm import tqdm
import json
import re
import numpy as np
from utils import *
from cifar_inference import load_dinov2, load_svm, run_cifar_inference
from cnn_inference import load_densenet, load_resnet, run_cnn_inference
from clip_inference import run_clip
from constants import artifact_index_dict, cifar_class_dict
from combine_algorithm import run_combination_algorithm
from gatekeeper import load_gatekeeper, run_gatekeeper

def main(folder_path, tsv_file_path, svm_path, cnn_models_base_path, gatekeeper_path, num_preds, limit_flag, clip_list_limit):
    # 1. Load detection models (NO OVIS HERE)
    dataframe = pd.read_csv(tsv_file_path, sep="\t")
    dinov2_model = load_dinov2()
    svm_model = load_svm(svm_path)
    resnet_dict = load_resnet(cnn_models_base_path)
    densenet_dict = load_densenet(cnn_models_base_path)
    gatekeeper_model = load_gatekeeper(gatekeeper_path)

    intermediate_data = []

    # 2. Process images
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # PHASE 1: THE GATEKEEPER
        is_fake, fake_prob = run_gatekeeper(img_path, gatekeeper_model)
        
        if not is_fake:
            print(f"\n[INFO] {filename} is REAL ({100 - (fake_prob*100):.1f}% confidence).")
            intermediate_data.append({
                "img_path": img_path,
                "filename": filename,
                "prediction": "Real",
                "fake_prob": fake_prob,
                "predicted_artifact_list": [],
                "display_name": "None"
            })
            with open("intermediate_results.json", "w") as f:
                json.dump(intermediate_data, f, indent=4)
            continue

        # PHASE 2: ARTIFACT EXTRACTION (Only for Fakes)
        print(f"\n[WARNING] {filename} is FAKE ({fake_prob*100:.1f}% confidence). Extracting artifacts...")
        detected_name = run_cifar_inference(img_path, dinov2_model, svm_model)
        
        object_class = None
        for idx, name in cifar_class_dict.items():
            if str(name).lower() == str(detected_name).lower():
                object_class = idx
                break
        
        if object_class is None:
            display_name = "Subject"
            lookup_index = 0 
        else:
            display_name = cifar_class_dict[object_class]
            lookup_index = object_class

        cnn_artifacts_dict = cnn_parser(dataframe["CNN"][lookup_index])
        cnn_models_dict = resnet_dict | densenet_dict
        cnn_output_dict = run_cnn_inference(img_path, cnn_models_dict, cnn_artifacts_dict)

        clip_prediction = run_clip(img_path, lookup_index, dataframe, clip_list_limit, 0)
        prediction_list = run_combination_algorithm(dataframe, lookup_index, clip_prediction, cnn_output_dict, num_preds, limit_flag)
        predicted_artifact_list = [key for key, value in artifact_index_dict.items() if value in prediction_list]

        intermediate_data.append({
            "img_path": img_path,
            "filename": filename,
            "prediction": "Fake",
            "fake_prob": fake_prob,
            "predicted_artifact_list": predicted_artifact_list,
            "display_name": display_name
        })

        with open("output.json", "w") as f:
            json.dump(intermediate_data, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Deepfake Artifact Pipeline - Part 1")
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--tsv_file_path', type=str, required=True)
    parser.add_argument('--svm_path', type=str, required=True)
    parser.add_argument('--cnn_model_base_path', type=str, required=True)
    parser.add_argument('--gatekeeper_path', type=str, default="./gatekeeper_model.pth")
    parser.add_argument('--num_preds', type=int, default=12)
    parser.add_argument('--limit_flag', type=int, default=0)
    parser.add_argument('--clip_list_limit', type=int, default=15)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.folder_path, args.tsv_file_path, args.svm_path, args.cnn_model_base_path, args.gatekeeper_path, args.num_preds, args.limit_flag, args.clip_list_limit)