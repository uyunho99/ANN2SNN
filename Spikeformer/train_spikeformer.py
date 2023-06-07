import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Spikformer
from model_spikeformer_time import SpikingViT

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

train_set = torchvision.datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1)

test_set = torchvision.datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# spiking_vit = Spikformer(img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_heads=8, 
#                 mlp_ratio=4., qkv_bias=False, dropout=0.1, num_classes=10, 
#                 time_steps=4, drop_path_rate=0.1, depths=10, sr_ratios=[8, 4, 2]).to(device)

spiking_vit = SpikingViT(img_size=32, patch_size=4, in_channels=3, embed_dim=192, num_heads=1,
                    mlp_ratio=4., qkv_bias=False, dropout=0.1, num_classes=10,
                    time_steps=4, drop_path_rate=0.1, depths=3, sr_ratios=[8, 4, 2]).to(device)

if torch.cuda.device_count() > 1:
    spiking_vit = nn.DataParallel(spiking_vit).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(spiking_vit.parameters(), lr=0.01, weight_decay=0.0001) # weight_decay: L2 규제
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1) # 학습률 감소
scaler = torch.cuda.amp.GradScaler() # Mixed Precision Training

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0.0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss.backward()
        loss.backward(retain_graph=True) # retain_graph: backward()를 여러번 호출할 때, 그래프를 유지
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()

        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct_predictions / len(loader.dataset)
    
    return epoch_loss, epoch_acc

def test(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct_predictions / len(loader.dataset)
    
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    epochs = 50
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(spiking_vit, train_loader, device, criterion, optimizer)
        print(f"Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}")

        test_loss, test_acc = test(spiking_vit, test_loader, device, criterion)
        print(f"Testing Loss: {test_loss:.4f}, Testing Acc: {test_acc:.4f}")
        
        scheduler.step()
