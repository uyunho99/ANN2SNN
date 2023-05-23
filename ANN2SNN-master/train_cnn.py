import torchvision
import torchvision.transforms as transforms
import time
import torch as th
import torch.nn as nn
import model_cnn as models

from torch.utils.data import DataLoader
from torch.backends.mps import is_available

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def cifar10_transform():
    return transforms.Compose([transforms.ToTensor()])

def mnist_transform():
    return transforms.Compose([transforms.ToTensor()])

def get_data_loaders(train_batch_size, data_name='cifar10'):

    if data_name == 'cifar10':
        tv_datasets = torchvision.datasets.CIFAR10
        tv_transform = cifar10_transform()
    elif data_name == 'mnist':
        tv_datasets = torchvision.datasets.MNIST
        tv_transform = mnist_transform()

    root_path = os.path.join(FILE_PATH, 'Dataset', data_name)

    train_dataset = tv_datasets(root=root_path, train=True, download=True, transform=tv_transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

    val_dataset = tv_datasets(root=root_path, train=False, download=False, transform=tv_transform)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=1)
    
    return train_loader, val_loader

def train_model(train_loader: DataLoader, ANN: nn.Module, optimizer: nn.Module, criterion: nn.Module, device: str):
    ANN.train()
    running_loss = 0
    for inputs, targets in train_loader:
        ANN.zero_grad()
        optimizer.zero_grad()
        
        # labels_ = th.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1).to(device)
        # targets = targets.to(device)
        inputs = inputs.float().to(device)
        outputs = ANN(inputs)
        labels = th.zeros(outputs.shape).scatter_(1, targets.view(-1, 1), 1).to(device)   # one-hot encoding
        loss = criterion(outputs, labels)
        running_loss += loss.cpu().item()
        loss.backward()
        optimizer.step()
    return running_loss

def test_model(loader: nn.Module, model: nn.Module, criterion: nn.Module, device: str):
    model.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for inputs, targets in loader:
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            labels = th.zeros(outputs.shape).scatter_(1, targets.view(-1, 1), 1).to(device)   # one-hot encoding
            loss = criterion(outputs, labels)
            # batch_size = inputs.size(0)
            # labels = th.zeros_like(inputs).scatter_(1, targets.view(-1, 1), 1).to(device)
            predicted = outputs.argmax(dim=-1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets.to(device)).sum().cpu().item())
    return correct / total


if __name__=='__main__':

    """ Hyper-parameters"""
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    num_epochs = 50

    """ Data Loader"""
    train_loader, val_loader = get_data_loaders(train_batch_size=128)
    
    """ Model """
    # ANN = models.Network_ANN()
    SNN = models.Network_SNN(time_window=40, threshold=1.0, max_rate=400)
    # ANN.to(device)
    SNN.to(device)
    # optimizer = th.optim.Adam(ANN.parameters(), lr=learning_rate, weight_decay=5e-5)
    # scheduler = th.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch)
    criterion = nn.MSELoss().to(device)

    """ Training Loop"""
    # for epoch in range(num_epochs):
    #     start_time = time.time()
        
    #     loss = train_model(train_loader, ANN, optimizer, criterion, device)
    #     scheduler.step()
        
    #     print(f"[{epoch+1}/{num_epochs}] LR: {scheduler.get_last_lr()[0]:.4e} "
    #           f"Training Loss: {loss:.5f} Time elasped:{time.time()-start_time:.2f} s")
        
    #     acc_ann = test_model(val_loader, ANN, device)
    #     print(f"ANN Test Accuracy: {100 * acc_ann:.3f}")

    # th.save(ANN.state_dict(), os.path.join(FILE_PATH, 'Model', 'ANN.pth'))
    ANN = th.load(os.path.join(FILE_PATH, 'Model', 'ANN.pth'))
    start_time = time.time()
    SNN.load_state_dict(ANN)
    for _ in range(30):
        acc_snn = test_model(val_loader, SNN, criterion, device)
        print(f"SNN Test Accuracy: {100 * acc_snn:.3f} Time elasped:{time.time()-start_time:.2f} s")

    # ANN.normalize_nn(train_loader)
    # for i, value in enumerate(ANN.factor_log):
    #     print('Normalization Factor for Layer %d: %3.5f' % (i, value))

    # time_window: This parameter typically refers to the length of the time period over which spikes are integrated or observed. 
    # It's a fundamental aspect of SNNs because unlike traditional artificial neurons that output a continuous value, spiking neurons fire binary spikes over time. 
    # The time window parameter is used to define the time frame for processing these spikes.
    # 
    # max_rate: In the context of SNNs, this parameter generally refers to the maximum firing rate of a neuron, which is the highest number of spikes that a neuron can emit in a given unit of time.
    # It's a way to constrain the activity of the neurons in the network, which can be important for controlling the dynamics of the network and preventing runaway feedback loops.

    # SNN_normalized = models.Network_SNN(time_window=40, threshold=1.0, max_rate=1000)
    # SNN_normalized.to(device)
    # SNN_normalized.load_state_dict(ANN.state_dict())

    # acc_snn_normalized = test_model(val_loader, SNN_normalized, criterion, device)
    # print(f"Normalized SNN Test Accuracy: {100 * acc_snn_normalized:.3f}")
