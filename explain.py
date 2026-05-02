import os
# SABSE PEHLE YE: Taaki koi bhi library load hone se pehle path set ho jaye
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"

import json
# Baaki imports iske baad
from ovis_inference import load_ovis
from artifact_explanation import artifact_explainer
from convert_to_json import parse_artifacts

def main():
    print("\n[INFO] Loading Ovis1.6-Gemma2-9B...")
    ovis, text_tokenizer, visual_tokenizer = load_ovis()
    
    if not os.path.exists("intermediate_results.json"):
        print("Error: intermediate_results.json not found. Run main.py first.")
        return

    with open("intermediate_results.json", "r") as f:
        data = json.load(f)
        
    final_output = []
    
    for item in data:
        filename = item['filename']
        prediction = item['prediction']
        fake_prob = item['fake_prob']
        
        print(f"\nProcessing: {filename} ({prediction})")
        
        if prediction == "Real":
            parsed_data = "No generative artifacts detected. Image passes structural and pixel-level authenticity checks."
        else:
            # Ovis generates explanation for the fakes
            ovis_output = artifact_explainer(
                item['img_path'], 
                item['display_name'], 
                item['predicted_artifact_list'], 
                len(item['predicted_artifact_list']), 
                ovis, 
                text_tokenizer, 
                visual_tokenizer,
                prediction 
            )
            parsed_data = parse_artifacts(ovis_output)
        
        entry = {
            "index": filename.split('.')[0], 
            "prediction": prediction,
            "confidence": f"{fake_prob*100:.1f}%" if prediction == "Fake" else f"{100 - (fake_prob*100):.1f}%",
            "explanation": parsed_data
        }
        final_output.append(entry)

        # Save to the final output file
        with open("output.json", "w") as json_file:
            json.dump(final_output, json_file, indent=4)
            
    print("\n[SUCCESS] Pipeline complete! Results saved to output.json")

if __name__ == '__main__':
    main()