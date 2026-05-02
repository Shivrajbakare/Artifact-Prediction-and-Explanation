# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from PIL import Image
# import cv2
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.datasets as datasets
# from torchvision.models import resnet50
# from torchvision.models import densenet121
# import os
# import timm
# from torchvision import transforms as T
# from sklearn.svm import SVC
# from joblib import load

# def load_dinov2():
#     """
#     Loads the DinoV2 model for image class inference.

#     Returns:
#         torch.nn.Module: The pre-trained ViT model ready for inference.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     MODEL = 'vit_small_patch14_dinov2.lvd142m'
#     model = timm.create_model(MODEL, pretrained=True, num_classes=0)
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     return model

# def load_svm(svm_path):
#     """
#     Loads a pre-trained Support Vector Machine (SVM) model from a file.

#     Args:
#         svm_path (str): The file path to the serialized SVM model (e.g., a `.joblib` file).

#     Returns:
#         sklearn.svm.SVC: The loaded SVM model, ready for inference or training.
#     """
#     svm = load(svm_path)
#     return svm


# def run_cifar_inference(image_path,model,svm):
#     """
#     Run an inference using the trained model to predict among the 10 CIFAR classes using a ViT model 
#     for feature extraction and an SVM model for classification.

#     Args:
#         image_path (str): Path to the input image file.
#         model (torch.nn.Module): A pre-trained Vision Transformer (ViT) model.
#         svm (sklearn.svm.SVC or similar): A trained Support Vector Machine (SVM) model.

#     Returns:
#         int: The predicted class label (1-based index) for the input image.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     data_config = timm.data.resolve_model_data_config(model)
#     transforms = timm.data.create_transform(**data_config, is_training=False)

#     # Load and preprocess the image
#     img = Image.open(image_path).convert("RGB")
#     img_transformed = transforms(img).unsqueeze(0).to(device)

#     # Extract features
#     with torch.no_grad():
#         feature = model.forward_features(img_transformed).cpu().numpy()
#         feature = feature.reshape(1, -1)  # Reshape for SVM

#     # Predict the label using the SVM
#     label = svm.predict(feature)
#     return label[0]

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.models import resnet50
from torchvision.models import densenet121
import os
import timm
from torchvision import transforms as T
from sklearn.svm import SVC
from joblib import load

def load_dinov2():
    """
    Loads the DinoV2 model for image class inference.

    Returns:
        torch.nn.Module: The pre-trained ViT model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = 'vit_small_patch14_dinov2.lvd142m'
    model = timm.create_model(MODEL, pretrained=True, num_classes=0)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def load_svm(svm_path):
    """
    Loads a pre-trained Support Vector Machine (SVM) model from a file.

    Args:
        svm_path (str): The file path to the serialized SVM model (e.g., a `.joblib` file).

    Returns:
        sklearn.svm.SVC: The loaded SVM model, ready for inference or training.
    """
    svm = load(svm_path)
    return svm


def run_cifar_inference(image_path,model,svm):
    """
    Run an inference using the trained model to predict among the 10 CIFAR classes using a ViT model 
    for feature extraction and an SVM model for classification.

    Args:
        image_path (str): Path to the input image file.
        model (torch.nn.Module): A pre-trained Vision Transformer (ViT) model.
        svm (sklearn.svm.SVC or similar): A trained Support Vector Machine (SVM) model.

    Returns:
        int: The predicted class label (1-based index) for the input image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Change this line:
        # Instead of getting everything, we just take the summary vector
        features = model(img_t)
        
        # If model(img_t) returns a tensor directly (common with torch.hub):
        # We ensure it's the shape (1, 384)
        if hasattr(features, 'last_hidden_state'):
            # If using HuggingFace version
            feat = features.last_hidden_state[:, 0, :].cpu().numpy()
        else:
            # If using torch.hub version
            feat = features.cpu().numpy()
            
    # Reshape to 2D for SVM (1 sample, 384 features)
    feat = feat.reshape(1, -1) 
    
    # Now predict
    prediction = svm.predict(feat)[0]
    return prediction