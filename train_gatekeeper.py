import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def train_gatekeeper():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare the Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading dataset from ./gatekeeper_dataset ...")
    dataset = datasets.ImageFolder(root='./gatekeeper_dataset', transform=transform)
    
    # Confirm PyTorch mapped them correctly!
    print(f"Class mapping: {dataset.class_to_idx}") 

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # 2. Build the Model (ResNet18)
    model = models.resnet18(weights='IMAGENET1K_V1') # Start with pre-trained vision weights
    num_ftrs = model.fc.in_features
    # Change final layer to output 1 value (Binary: 0 or 1)
    model.fc = nn.Linear(num_ftrs, 1) 
    model = model.to(device)

    # 3. Loss and Optimizer
    # BCEWithLogitsLoss is required for binary classification with a single output node
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (Let's do 5 epochs to start)
    num_epochs = 5
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for a nice progress bar in your VS Code terminal
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            # Reshape labels to match output format
            labels = labels.float().unsqueeze(1).to(device) 

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f}")

    # 5. Save the final weights!
    print("Saving model weights...")
    torch.save(model.state_dict(), 'gatekeeper_model.pth')
    print("SUCCESS! Saved as gatekeeper_model.pth")

if __name__ == '__main__':
    train_gatekeeper()