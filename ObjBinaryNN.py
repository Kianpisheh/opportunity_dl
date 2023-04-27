from torch import nn


class ObjBinaryNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 3, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(3*20, 4)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.fc(x)
        x = self.relu2(x)
        x = self.softmax(x)
        
        return x
        