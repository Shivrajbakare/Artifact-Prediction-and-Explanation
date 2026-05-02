import torch
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from sklearn.svm import SVC
import joblib
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
dinov2.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_svm():
    df = pd.read_csv('trainLabels.csv')
    
    # We will check both possible locations for the images
    possible_paths = ['./train', './train/train']
    
    features = []
    labels = []
    
    print("Searching for images...")
    
    processed_count = 0
    # Let's try to process 1,000 images to get your .joblib file quickly
    limit = 200 

    for idx, row in tqdm(df.iterrows(), total=limit):
        if processed_count >= limit:
            break
            
        img_found = False
        for p in possible_paths:
            img_path = os.path.join(p, f"{row['id']}.png")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = dinov2(img_t).cpu().numpy().reshape(-1)
                        features.append(feat)
                        labels.append(row['label'])
                    img_found = True
                    processed_count += 1
                    break # Stop looking in other paths if found
                except Exception as e:
                    continue
    
    if len(features) == 0:
        print("\n!!! ERROR: STILL NO IMAGES FOUND !!!")
        print(f"I checked: {[os.path.abspath(p) for p in possible_paths]}")
        print("Please ensure your .png files are in one of those folders.")
        return

    print(f"\nSuccessfully extracted {len(features)} images.")
    print("Training SVM (This might take a minute)...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(np.array(features), labels)

    print("Saving model...")
    joblib.dump(clf, 'svm_cifar10_model.joblib')
    print("SUCCESS! svm_cifar10_model.joblib has been created.")

if __name__ == "__main__":
    train_svm()