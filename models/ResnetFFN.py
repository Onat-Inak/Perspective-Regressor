import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


class ResnetFFN(nn.Module):
    def __init__(self, device, num_outputs, batch_first, conv_channel, fc_hidden_dims):
        super(ResnetFFN, self).__init__()
        self.device = device
        self.num_outputs = num_outputs
        self.batch_first = batch_first
        self.conv_channel = conv_channel
        self.fc_hidden_dims = fc_hidden_dims

        # initialize Resnet50 weigths and prepocessing images
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms()

        # initialize Resnet50 backbone by removing the last layer
        self.backbone = torch.nn.Sequential(
            *(list(resnet50(weights=self.weights).children())[:-1])
        ).to(self.device)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # 1x1 convolution layer to reduce channel size of Resnet50, (last channel dim is 2048):
        self.conv1x1 = nn.Conv2d(
            2048, self.conv_channel, kernel_size=1, stride=1, padding=0
        ).to(self.device)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_channel, self.fc_hidden_dims[0], device=self.device),
            nn.ReLU(),
            nn.Linear(
                self.fc_hidden_dims[0], self.fc_hidden_dims[1], device=self.device
            ),
            nn.ReLU(),
            nn.Linear(
                self.fc_hidden_dims[1], self.fc_hidden_dims[2], device=self.device
            ),
            nn.ReLU(),
            nn.Linear(self.fc_hidden_dims[2], self.num_outputs, device=self.device),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x):

        x = x.to(self.device)
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)

        return x
