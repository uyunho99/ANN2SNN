import torchvision
import torchvision.transforms as transforms
import time
import torch
import torch.nn as nn
import model_cnn as models
from torch.backends.mps import is_available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 50

ANN = models.Network_ANN()
SNN = models.Network_SNN(time_window=40, threshold=1.0, max_rate=400)
ANN.to(device)
SNN.to(device)
optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch)
criterion = nn.MSELoss().to(device)

def cifar10_transform():
    return transforms.Compose([transforms.ToTensor()])

def get_data_loaders(train_batch_size):
    train_dataset = torchvision.datasets.CIFAR10(
        root="ANN2SNN-master/Dataset/cifar10", train=True, download=True, transform=cifar10_transform())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = torchvision.datasets.CIFAR10(
        root="ANN2SNN-master/Dataset/cifar10", train=False, download=False, transform=cifar10_transform())
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False)
    
    return train_loader, val_loader

def train_model(train_loader, ANN, optimizer, criterion, device):
    ANN.train()
    running_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        batch_sz = inputs.size(0)
        ANN.zero_grad()
        optimizer.zero_grad()
        inputs = inputs.float().to(device)
        labels_ = torch.zeros(batch_sz, 10).scatter_(
            1, targets.view(-1, 1), 1).to(device)
        outputs = ANN(inputs)
        loss = criterion(outputs, labels_)
        running_loss += loss.cpu().item()
        loss.backward()
        optimizer.step()
    return running_loss

def test_model(loader, model, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_sz = inputs.size(0)
            inputs = inputs.float().to(device)
            labels_ = torch.zeros(batch_sz, 10).scatter_(
                1, targets.view(-1, 1), 1).to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            loss = criterion(outputs, labels_)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().cpu().item())
    return correct / total

train_loader, val_loader = get_data_loaders(train_batch_size=100)

for epoch in range(num_epochs):
    start_time = time.time()
    
    loss = train_model(train_loader, ANN, optimizer, criterion, device)
    scheduler.step()
    
    print('Epoch [%d/%d], Training Loss: %.5f Time elasped:%.2f s' 
          % (epoch+1, num_epochs, loss, time.time()-start_time))
    
    acc_ann = test_model(val_loader, ANN, criterion, device)
    print("ANN Test Accuracy: %.3f" % (100 * acc_ann))
    
    # SNN.load_state_dict(ANN.state_dict())
    # acc_snn = test_model(val_loader, SNN, criterion, device)
    # print("SNN Test Accuracy: %.3f Time elasped:%.2f s" %
    #       (100 * acc_snn, time.time()-start_time))

# ANN.normalize_nn(train_loader)
# for i, value in enumerate(ANN.factor_log):
#     print('Normalization Factor for Layer %d: %3.5f' % (i, value))

# time_window: This parameter typically refers to the length of the time period over which spikes are integrated or observed. 
# It's a fundamental aspect of SNNs because unlike traditional artificial neurons that output a continuous value, spiking neurons fire binary spikes over time. 
# The time window parameter is used to define the time frame for processing these spikes.
# 
# max_rate: In the context of SNNs, this parameter ge\[nerally refers to the maximum firing rate of a neuron, which is the highest number of spikes that a neuron can emit in a given unit of time.
# It's a way to constrain the activity of the neurons in the network, which can be important for controlling the dynamics of the network and preventing runaway feedback loops.

torch.save(ANN.state_dict(), './Model/ANN_cifar10.pth')

# SNN_normalized = models.Network_SNN(
#     time_window=40, threshold=1.0, max_rate=1000)
# SNN_normalized.to(device)
# SNN_normalized.load_state_dict(ANN.state_dict())

# acc_snn_normalized = test_model(val_loader, SNN_normalized, criterion, device)
# print("Normalized SNN Test Accuracy: %.3f" % (100 * acc_snn_normalized))