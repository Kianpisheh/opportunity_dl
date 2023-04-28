import torch
from torch import nn

class ObjBinaryNN(nn.Module):
    def __init__(self):
        super().__init__()

        n_input_feature = 20
        n_channel_conv = 5
        n_lstm_hidden = 11


        self.conv = nn.Conv1d(n_input_feature, n_channel_conv, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=n_channel_conv, hidden_size=n_lstm_hidden, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(n_lstm_hidden, 4)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x, h = self.lstm(x.transpose(1,2))
        x = self.fc1(x[:,-1])
        x = self.relu2(x)
        x = self.softmax(x)
        
        return x
        