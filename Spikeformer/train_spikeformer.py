import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import model_spikeformer as model
import model_spikeformer_time as model

# torch.autograd.set_detect_anomaly(True)

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

train_set = torchvision.datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)

test_set = torchvision.datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1)

# Initialize the model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SpikingViT = model.SpikingViT().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(SpikingViT.parameters(), lr=0.01, weight_decay=0.0001) # weight_decay: L2 규제
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1) # 학습률 감소

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0.0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # loss.backward(retain_graph=True) # retain_graph: backward()를 여러번 호출할 때, 그래프를 유지
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()

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

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct_predictions / len(loader.dataset)
    
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    epochs = 50
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(SpikingViT, train_loader, device, criterion, optimizer)
        print(f"Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}")

        test_loss, test_acc = test(SpikingViT, test_loader, device, criterion)
        print(f"Testing Loss: {test_loss:.4f}, Testing Acc: {test_acc:.4f}")
        
        scheduler.step()

