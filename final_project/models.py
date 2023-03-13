import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(MLP, self).__init__()

        input_size_1d = input_size[0] * input_size[1]

        self.layers = nn.Sequential(
            nn.Linear(input_size_1d, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# We implement the network in this paper:
# https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(CNN, self).__init__()
        F1 = 8
        D = 2
        F2 = F1 * D
        F3 = F2
        SampleRate = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, SampleRate), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(22, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=F2, out_channels=F3, kernel_size=(1, SampleRate // 4), bias=False
            ),
            nn.Conv2d(in_channels=F3, out_channels=F3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(),
        )

        w1 = int((input_size[1] - SampleRate + 1) / 1)
        w2 = int(w1 / 2)
        w3 = w2 - SampleRate // 4 + 1
        w4 = int(w3 / 4)

        self.fc = nn.Linear(F3 * 1 * w4, output_size)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# We implement and modify the network based on the paper:
# https://www.sciencedirect.com/science/article/pii/S016502702100217X?via%3Dihub
class CRNN(nn.Module):
    def __init__(self, output_size, use_attention, rnn_hidden_size=128, num_rnn_layers=3):
        super(CRNN, self).__init__()

        self.use_attention = use_attention
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_layers = num_rnn_layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 10)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(21, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 10)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 10)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(),
        )

        self.fc1 = nn.Linear(256, 256)

        self.rnn = nn.LSTM(
            256,
            rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=False,
        )
        if use_attention:
            self.fc_attn = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size * 2)

        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(rnn_hidden_size * 2, output_size)

    def forward(self, x):
        # CNN
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.transpose(1, 3).flatten(start_dim=2)

        # lstm
        x = self.fc1(x).transpose(0, 1)
        rnn_output, (hiddens, _) = self.rnn(x)
        rnn_output = rnn_output.permute(1, 0, 2)
        final_hidden = hiddens.view(self.num_rnn_layers, 2, -1, self.rnn_hidden_size)[-1]
        final_hidden = torch.cat((final_hidden[0], final_hidden[1]), 1)

        # attention
        if self.use_attention:
            attn_weights = self.fc_attn(rnn_output)
            attn_weights = torch.bmm(attn_weights, final_hidden.unsqueeze(2))
            attn_weights = F.softmax(attn_weights.squeeze(2), 1)
            final_hidden = torch.bmm(rnn_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(
                2
            )

        # fc cls
        output = self.fc2(self.dropout(final_hidden))
        return output
