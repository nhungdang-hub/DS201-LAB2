import torch 
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2)
        )

        self.pooling_1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )

        self.conv_2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            padding=0
        )

        self.pooling_2 = nn.AvgPool2d(
            stride=(2, 2),
            kernel_size=(2, 2),
            padding=(0, 0)
        )

        self.conv_3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            padding=(0, 0)
        )

        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )
        
        self.output = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward(self, images: torch.Tensor):
        ''''
            images (bs, 28, 28)
        '''

        images = images.unsqueeze(1) # (bs, 1, 28, 28)                                                                                                                                                                                

        features = F.sigmoid(self.conv_1(images)) # (bs, 6, 28, 28)
        features = self.pooling_1(features) # (bs, 6, 14, 14)

        features = F.sigmoid(self.conv_2(features)) # (bs, 16, 10, 10)
        features = self.pooling_2(features) # (bs, 16, 5, 5)

        features = F.sigmoid(self.conv_3(features)) # (bs, 120, 1, 1)
        features = features.squeeze(-1).squeeze(-1) # (bs, 120)                                                                                                                                                                        
        
        features = F.sigmoid(self.fc(features)) # (bs, 84)
        logits = self.output(features) # (bs, 10)

        return logits

