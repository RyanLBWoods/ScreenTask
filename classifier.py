import torch.nn as nn


class simpleClassifier(nn.Module):
    """
    Simple ConvNet for classification
    """
    def __init__(self):
        super(simpleClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 5,),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(20 * 10 * 10, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        in_size = x.shape[0]
        x = self.conv(x)
        x = x.view(in_size, -1)
        x = self.dense(x)
        return x
