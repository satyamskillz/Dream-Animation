import torch
from torchvision import models


class net(torch.nn.Module):
    def __init__(self):
        self.vgg = models.vgg19(pretrained=True).features
        for para in self.vgg.parameters():
            para.requires_grad_(False)
