import torchvision
import torchvision.transforms as transforms
import time
import torch
import torch.nn as nn
import torch.optim as optim
import model_cnn as models
from torch.backends.mps import is_available


def mnist_transform():
    return transforms.Compose([transforms.ToTensor()])

def cifar_transform():
    return transforms.Compose([transforms.ToTensor()])


train_batch_size = 100
train_dataset = torchvision.datasets.CIFAR10(root="./Dataset/cifar10", train=True, download=True, transform=cifar_transform())
val_dataset = torchvision.datasets.CIFAR10(root="./Dataset/cifar10", train=False, download=False, transform=cifar_transform())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=False)

# device = torch.device("cpu" if not is_available() else "mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-3
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 20
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

ANN = models.Network_ANN()
SNN = models.Network_SNN(time_window=40, threshold=1.0, max_rate=400)
ANN.to(device)
SNN.to(device)
# optimizer = torch.optim.SGD(ANN.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch)
criterion = nn.MSELoss().to(device)

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        batch_sz = inputs.size(0)
        ANN.zero_grad()
        optimizer.zero_grad()
        inputs = inputs.float().to(device)
        current_batch_size = inputs.size(0)
        labels_ = torch.zeros(current_batch_size, 10).scatter_(
            1, targets.view(-1, 1), 1).to(device)
        outputs = ANN(inputs)
        loss = criterion(outputs, labels_)
        running_loss += loss.cpu().item()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], LR: %.4f Training Loss: %.5f Time elasped:%.2f s'
                  % (epoch+1, num_epochs, i+1, len(train_dataset)//train_batch_size, scheduler.get_lr()[0], running_loss/100, time.time()-start_time))
            running_loss = 0
    scheduler.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_sz = inputs.size(0)
            inputs = inputs.float().to(device)
            labels_ = torch.zeros(batch_sz, 10).scatter_(
                1, targets.view(-1, 1), 1).to(device)
            outputs = ANN(inputs)
            targets = targets.to(device)
            loss = criterion(outputs, labels_)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().cpu().item())
    print("ANN Test Accuracy: %.3f" % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

    correct = 0
    total = 0

# plt.title("Loss Curve")
# plt.plot(loss_train_record, label="Train Loss")
# plt.plot(loss_test_record, label="Test Loss")
# plt.legend(loc='upper right')
# plt.savefig("loss_curve.png")

    SNN.load_state_dict(ANN.state_dict())
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_sz = inputs.size(0)
        targets = targets.to(device)
        inputs = inputs.float().to(device)
        outputs = SNN(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().cpu().item())
    print("SNN Test Accuracy: %.3f" % (100 * correct / total))

with open('ANN_acc_record.txt', 'w') as f:
    for item in acc_record:
        f.write("%s\n" % item)
with open('ANN_loss_train_record.txt', 'w') as f:
    for item in loss_train_record:
        f.write("%s\n" % item)
with open('ANN_loss_test_record.txt', 'w') as f:
    for item in loss_test_record:
        f.write("%s\n" % item)

ANN.normalize_nn(train_loader)
for i, value in enumerate(ANN.factor_log):
    print('Normalization Factor for Layer %d: %3.5f' % (i, value))
    
correct = 0
total = 0
SNN_normalized = models.Network_SNN(
    time_window=40, threshold=1.0, max_rate=1000)
SNN_normalized.to(device)
SNN_normalized.load_state_dict(ANN.state_dict())
for batch_idx, (inputs, targets) in enumerate(val_loader):
    batch_sz = inputs.size(0)
    targets = targets.to(device)
    inputs = inputs.float().to(device)
    outputs = SNN_normalized(inputs)
    _, predicted = outputs.max(1)
    total += float(targets.size(0))
    correct += float(predicted.eq(targets).sum().cpu().item())
print("Normalized SNN Test Accuracy: %.3f" % (100 * correct / total))
