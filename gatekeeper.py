import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

def load_gatekeeper(model_path):
    """
    Loads a lightweight CNN (e.g., ResNet18) trained for Real/Fake detection.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assuming you trained a ResNet18 for binary classification
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Single output node for binary classification
    
    # Load your trained weights (uncomment and use this once you train the model)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    return model

def run_gatekeeper(image_path, model):
    """
    Analyzes an image and returns True if Fake, False if Real.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        # Convert raw output to a probability between 0 and 1
        probability = torch.sigmoid(output).item()
        
    # If probability > 50%, it's Fake
    is_fake = probability > 0.5
    return is_fake, probability