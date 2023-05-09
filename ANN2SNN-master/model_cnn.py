import torchvision
import torchvision.transforms as transforms
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network_ANN(nn.Module):
    def __init__(self):
        super(Network_ANN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.7)
        self.subsample1 = nn.AvgPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=5,
                               stride=1,
                               padding=1,
                               bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.9)
        self.subsample2 = nn.AvgPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False)
        self.HalfRect3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.8)
        self.subsample3 = nn.AvgPool2d(2, 2, 0)
        self.fc1 = nn.Linear(128, 10, bias=False)
        self.HalfRect4 = nn.ReLU()

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, input):
        x = self.conv1(input)  # 1*28*28 -> 16*28*28
        x = self.HalfRect1(x)
        x = self.dropout1(x)
        x = self.subsample1(x)  # 16*28*28 -> 16*14*14
        x = self.conv2(x)  # 16*14*14 -> 32*14*14
        x = self.HalfRect2(x)
        x = self.dropout2(x)
        x = self.subsample2(x)  # 32*14*14 -> 32*7*7
        x = self.conv3(x)  # 32*7*7 -> 32*4*4
        x = self.HalfRect3(x)
        x = self.dropout3(x)
        x = self.subsample3(x)  # 32*4*4 -> 32*2*2
        x = x.view(-1, 128)  # 32*2*2 -> 256
        x = self.fc1(x)  # 256 -> 10
        x = self.HalfRect4(x)
        return x

    def normalize_nn(self, train_loader):
        conv1_weight_max = torch.max(F.relu(self.conv1.weight))
        conv2_weight_max = torch.max(F.relu(self.conv2.weight))
        conv3_weight_max = torch.max(F.relu(self.conv3.weight))
        fc1_weight_max = torch.max(F.relu(self.fc1.weight))
        conv1_activation_max = 0.0
        conv2_activation_max = 0.0
        conv3_activation_max = 0.0
        fc1_activation_max = 0.0

        self.eval()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            x = inputs.float().to(self.device)
            x = self.dropout1(self.HalfRect2(self.conv1(x)))
            conv1_activation_max = max(conv1_activation_max, torch.max(x))
            x = self.subsample1(x)
            x = self.dropout2(self.HalfRect2(self.conv2(x)))
            conv2_activation_max = max(conv2_activation_max, torch.max(x))
            x = self.subsample2(x)
            x = self.dropout3(self.HalfRect3(self.conv3(x)))
            conv3_activation_max = max(conv3_activation_max, torch.max(x))
            x = self.subsample3(x)
            x = x.view(-1, 128)
            x = self.HalfRect4(self.fc1(x))
            fc1_activation_max = max(fc1_activation_max, torch.max(x))
        self.train()

        self.factor_log = []
        previous_factor = 1

        scale_factor = max(conv1_weight_max, conv1_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.conv1.weight.data = self.conv1.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(conv2_weight_max, conv2_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.conv2.weight.data = self.conv2.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(conv3_weight_max, conv3_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.conv3.weight.data = self.conv3.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor

        scale_factor = max(fc1_weight_max, fc1_activation_max)
        applied_inv_factor = (scale_factor / previous_factor).item()
        self.fc1.weight.data = self.fc1.weight.data / applied_inv_factor
        self.factor_log.append(1 / applied_inv_factor)
        previous_factor = applied_inv_factor


class Network_SNN(nn.Module):
    def __init__(self, time_window=35, threshold=1.0, max_rate=200):
        super(Network_SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=False)
        self.HalfRect1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.7)
        self.subsample1 = nn.AvgPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=5,
                               stride=1,
                               padding=1,
                               bias=False)
        self.HalfRect2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.9)
        self.subsample2 = nn.AvgPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2,
                               padding=1,
                               bias=False)
        self.HalfRect3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.8)
        self.subsample3 = nn.AvgPool2d(2, 2, 0)
        self.fc1 = nn.Linear(128, 10, bias=False)
        self.HalfRect4 = nn.ReLU()

        self.threshold = threshold
        self.time_window = time_window
        self.dt = 0.001  # second
        self.refractory_t = 0
        self.max_rate = max_rate
        self.rescale_factor = 1.0 / (self.dt * self.max_rate)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def mem_update(self, t,  operator, mem, input_spk, leak=0, refrac_end=None):
        # Get input impulse from incoming spikes
        impulse = operator(input_spk)
        # Add input to membrane potential
        mem = mem + impulse + leak
        # Check for spiking
        output_spk = (mem >= self.threshold).float()
        # Reset
        mem = mem * (1. - output_spk)
        # Ban updates until....
        if refrac_end is not None:
            refrac_end = output_spk * (t + self.refractory_t)
        return mem, output_spk

    def forward(self, input):
        batch_size = input.size(0)

        spk_input = spksum_input = torch.zeros(
            batch_size, 1, 28, 28, device=self.device)
        mem_post_conv1 = spk_post_conv1 = spksum_post_conv1 = torch.zeros(
            batch_size, 16, 28, 28, device=self.device)
        mem_post_subsample1 = spk_post_subsample1 = spksum_post_subsample1 = torch.zeros(
            batch_size, 16, 14, 14, device=self.device)
        mem_post_conv2 = spk_post_conv2 = spksum_post_conv2 = torch.zeros(
            batch_size, 32, 14, 14, device=self.device)
        mem_post_subsample2 = spk_post_subsample2 = spksum_post_subsample2 = torch.zeros(
            batch_size, 32, 7, 7, device=self.device)
        mem_post_conv3 = spk_post_conv3 = spksum_post_conv3 = torch.zeros(
            batch_size, 32, 4, 4, device=self.device)
        mem_post_subsample3 = spk_post_subsample3 = spksum_post_subsample3 = torch.zeros(
            batch_size, 32, 2, 2, device=self.device)
        mem_post_fc1 = spk_post_fc1 = spksum_post_fc1 = torch.zeros(
            batch_size, 10, device=self.device)

        for t in range(self.time_window):
            spk_input = (torch.rand(input.size(), device=self.device)
                         * self.rescale_factor <= input).float()
            spksum_input = spksum_input + spk_input

            mem_post_conv1, spk_post_conv1 = self.mem_update(
                t, self.conv1, mem_post_conv1, spk_input)
            spksum_post_conv1 = spksum_post_conv1 + spk_post_conv1

            mem_post_subsample1, spk_post_subsample1 = self.mem_update(
                t, self.subsample1, mem_post_subsample1, spk_post_conv1)
            spksum_post_subsample1 = spksum_post_subsample1 + spk_post_subsample1

            mem_post_conv2, spk_post_conv2 = self.mem_update(
                t, self.conv2, mem_post_conv2, spk_post_subsample1)
            spksum_post_conv2 = spksum_post_conv2 + spk_post_conv2

            mem_post_subsample2, spk_post_subsample2 = self.mem_update(
                t, self.subsample2, mem_post_subsample2, spk_post_conv2)
            spksum_post_subsample2 = spksum_post_subsample2 + spk_post_subsample2

            mem_post_conv3, spk_post_conv3 = self.mem_update(
                t, self.conv3, mem_post_conv3, spk_post_subsample2)
            spksum_post_conv3 = spksum_post_conv3 + spk_post_conv3

            mem_post_subsample3, spk_post_subsample3 = self.mem_update(
                t, self.subsample3, mem_post_subsample3, spk_post_conv3)
            spksum_post_subsample3 = spksum_post_subsample3 + spk_post_subsample3

            spk_post_subsample3_ = spk_post_subsample3.view(batch_size, 128)
            mem_post_fc1, spk_post_fc1 = self.mem_update(
                t, self.fc1, mem_post_fc1, spk_post_subsample3_)
            spksum_post_fc1 = spksum_post_fc1 + spk_post_fc1
        outputs = spksum_post_fc1 / self.time_window
        return outputs


if __name__ == "__main__":
    from torch.autograd import Variable
    features = []

    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        features.append(output.data.cpu().numpy())
    net = Network_ANN()
    net.to(torch.device("cpu"))
    net.train()

    for w in net.named_parameters():
        print(w[0])

    for m in net.modules():
        m.register_forward_hook(hook)

    # (batch_size, channels, height, width)
    y = net(Variable(torch.randn(32, 1, 28, 28)))
    print(y.size())
