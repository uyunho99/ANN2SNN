# Spiking Vision Transformer

This repository contains an implementation of a Spiking Vision Transformer (SpikingViT), an innovative approach that merges the strengths of Vision Transformers (ViT) and Spiking Neural Networks (SNN) for image classification tasks. The model is trained on the CIFAR-10 dataset using PyTorch.

## Model Architecture

The model uses a modified version of the ViT that employs spiking neurons. A standard ViT begins by splitting the input images into patches, each of which is linearly transformed into a vector. These vectors are then treated as a sequence, which is processed by a series of transformer encoder layers that use multi-head self-attention mechanisms and MLPs. The output sequence is passed to a classification head to make predictions.

In the SpikingViT, we replace the traditional artificial neurons with spiking neurons, which only fire when the input exceeds a certain threshold, leading to potentially more energy-efficient neural networks.

Getting Started

Prerequisites
Python 3.x
PyTorch
torchvision
CUDA (if available)
spikingjelly==0.0.0.0.12
timm==0.5.4
