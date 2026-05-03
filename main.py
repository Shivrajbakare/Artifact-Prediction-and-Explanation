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
from clip_inference import load_clip, run_clip  # <--- CHANGED HERE
from convert_to_json import parse_artifacts, json_formatting
from constants import artifact_index_dict, cifar_class_dict
from combine_algorithm import run_combination_algorithm
from artifact_explanation import artifact_explainer
from ovis_inference import load_ovis

def main(folder_path, tsv_file_path, svm_path, cnn_models_base_path, num_preds, limit_flag, clip_list_limit):
    """
    Runs the full pipeline for artifact prediction and explanation.
    """
    # 1. Load all models cleanly
    print("[INFO] Loading all models...")
    dataframe = pd.read_csv(tsv_file_path, sep="\t")
    dinov2_model = load_dinov2()
    svm_model = load_svm(svm_path)
    resnet_dict = load_resnet(cnn_models_base_path)
    densenet_dict = load_densenet(cnn_models_base_path)
    
    clip_model = load_clip() # <--- NEW: Loading CLIP properly
    
    ovis, text_tokenizer, visual_tokenizer = load_ovis()

    json_output = []

    # 2. Process images
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # --- CLASS PREDICTION & MAPPING ---
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

        # --- ARTIFACT PREDICTION ---
        cnn_artifacts_dict = cnn_parser(dataframe["CNN"][lookup_index])
        cnn_models_dict = resnet_dict | densenet_dict
        cnn_output_dict = run_cnn_inference(img_path, cnn_models_dict, cnn_artifacts_dict)

        # <--- NEW: Passed clip_model into the function
        clip_prediction = run_clip(clip_model, img_path, lookup_index, dataframe, clip_list_limit, 0)

        # Combine predictions
        prediction_list = run_combination_algorithm(dataframe, lookup_index, clip_prediction, cnn_output_dict, num_preds, limit_flag)
        predicted_artifact_list = [key for key, value in artifact_index_dict.items() if value in prediction_list]

        # --- WEIGHTED REAL VS FAKE LOGIC ---
        strong_indicators = [
            "Asymmetric features in naturally symmetric objects",
            "Misshapen ears or appendages",
            "Incorrect Skin Tones",
            "Ghosting effects: Semi-transparent duplicates of elements",
            "Non-manifold geometries in rigid structures"
        ]
        
        total_score = 0
        for artifact in predicted_artifact_list:
            if artifact in strong_indicators:
                total_score += 2.5  
            else:
                total_score += 1.0  
        
        prediction_label = "Fake" if total_score > 5.5 else "Real"

        # --- EXPLANATION GENERATION ---
        print(f"\n[DEBUG] {filename} -> {display_name} ({prediction_label})")
        ovis_output = artifact_explainer(
            img_path, 
            display_name, 
            predicted_artifact_list, 
            len(predicted_artifact_list), 
            ovis, 
            text_tokenizer, 
            visual_tokenizer,
            prediction_label 
        )
        
        # --- OUTPUT FORMATTING ---
        parsed_data = parse_artifacts(ovis_output)
        
        entry = {
            "index": filename.split('.')[0], 
            "prediction": prediction_label,
            "explanation": parsed_data
        }
        
        json_output.append(entry)

        with open("output.json", "w") as json_file:
            json.dump(json_output, json_file, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Deepfake Artifact Pipeline")
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--tsv_file_path', type=str, required=True)
    parser.add_argument('--svm_path', type=str, required=True)
    parser.add_argument('--cnn_model_base_path', type=str, required=True)
    parser.add_argument('--num_preds', type=int, default=12)
    parser.add_argument('--limit_flag', type=int, default=0)
    parser.add_argument('--clip_list_limit', type=int, default=15)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.folder_path, args.tsv_file_path, args.svm_path, args.cnn_model_base_path, args.num_preds, args.limit_flag, args.clip_list_limit)
