from torch import nn
import torch


def pretrainedDNN(pred):
    if pred == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', weights="ResNet152_Weights.DEFAULT")
        model.fc = nn.Identity()
        model.eval()
    else:
        raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")
    return model