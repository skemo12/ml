from torch import nn


class ECG1DCNN(nn.Module):
    def __init__(self):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 43, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(64, 2)

        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc3(x)
        x = self.relu5(x)
        return x
