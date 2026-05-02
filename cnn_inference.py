import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from constants import densenet_artifacts, resnet_artifacts

def load_densenet(weights_base_path: str) -> dict:
    """Load the DenseNet models

    Args:
        weights_base_path (str): Folder Path where the State_dicts for the models are saved

    Returns:
        dict: Dictionary of artifact_idx and the respective models
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_dict=dict()

    for artifact_index in densenet_artifacts: 
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2) 
        weights_path = f"{weights_base_path}/densenet_{artifact_index}.pth"
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models_dict[artifact_index] = model

    return models_dict


def load_resnet(weights_base_path: str) -> dict:
    """Load the ResNet models

    Args:
        weights_base_path (str): Folder Path where the State_dicts for the models are saved

    Returns:
        dict: Dictionary of artifact_idx and the respective models
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_dict=dict()

    for artifact_index in resnet_artifacts: 
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2) 
        weights_path = f"{weights_base_path}/resnet_{artifact_index}.pth"
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models_dict[artifact_index] = model

    return models_dict


def run_cnn_inference(img_path: str,models_dict: dict, cnn_classes:dict) -> dict:
    """
    Run inference on a given image using a dictionary of ResNet and DenseNet models, with class predictions stored in a dictionary.

    Args:
        img_path (str): Path to the input image file.
        models_dict (dict): A dictionary where keys are model identifiers and values are the corresponding CNN models.
        cnn_classes (dict): A dictionary containing class specific artifacts with fine tuned CNN models.

    Returns:
        dict: A dictionary where keys are model identifiers and values are the predicted class labels (0 or 1) for the input image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),    
        transforms.ToTensor(),          
        transforms.Normalize(           
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(img_path).convert("RGB") 
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    predictions_dict=dict()
    
    # Runs ResNet or DenseNet based on which artifact is being tested
    with torch.no_grad():
        for artifact_idx, model in models_dict.items():
            if artifact_idx in cnn_classes.keys():
                output = model(input_tensor)
                _, predicted_class = torch.max(output, 1)
                predictions_dict[artifact_idx] = predicted_class.item()

    return predictions_dict