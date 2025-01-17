import torch.nn as nn
import torch.nn.functional as F


class SepConvNet(nn.Module):
    def __init__(self, t1, f1, t2, f2, N1, N2, input_shape, NN=128):
        super(SepConvNet, self).__init__()
        # (vconv): Conv2d(1, 50, kernel_size=(513/3, 1), stride=(1, 1))
        self.vconv = nn.Conv2d(1, N1, kernel_size=(int(f1 / 3), t1), padding=0)
        self.maxpool = nn.MaxPool2d((int(f1 / 3) * 2, 1), return_indices=True)
        # (hconv): Conv2d(50, 30, kernel_size=(1, 15), stride=(1, 1))
        self.hconv = nn.Conv2d(N1, N2, kernel_size=(f2, t2))

        # (fc0): Linear(in_features=1140, out_features=128, bias=True)
        self.fc0 = nn.Linear(N2 * (input_shape[0] - f1 - f2 + 2) * (input_shape[1] - t1 - t2 + 2), NN)
        # (fc1): Linear(in_features=128, out_features=1140, bias=True)
        self.fc1 = nn.Linear(NN, N2 * (input_shape[0] - f1 - f2 + 2) * (input_shape[1] - t1 - t2 + 2))
        self.fc2 = nn.Linear(NN, N2 * (input_shape[0] - f1 - f2 + 2) * (input_shape[1] - t1 - t2 + 2))
        self.fc3 = nn.Linear(NN, N2 * (input_shape[0] - f1 - f2 + 2) * (input_shape[1] - t1 - t2 + 2))
        self.fc4 = nn.Linear(NN, N2 * (input_shape[0] - f1 - f2 + 2) * (input_shape[1] - t1 - t2 + 2))

        # (hdeconv1): ConvTranspose2d(30, 50, kernel_size=(1, 15), stride=(1, 1))
        self.hdeconv1 = nn.ConvTranspose2d(N2, N1, kernel_size=(f2, t2))
        self.hdeconv2 = nn.ConvTranspose2d(N2, N1, kernel_size=(f2, t2))
        self.hdeconv3 = nn.ConvTranspose2d(N2, N1, kernel_size=(f2, t2))
        self.hdeconv4 = nn.ConvTranspose2d(N2, N1, kernel_size=(f2, t2))

        self.maxunpool1 = nn.MaxUnpool2d((int(f1 / 3) * 2, 1))
        self.maxunpool2 = nn.MaxUnpool2d((int(f1 / 3) * 2, 1))
        self.maxunpool3 = nn.MaxUnpool2d((int(f1 / 3) * 2, 1))
        self.maxunpool4 = nn.MaxUnpool2d((int(f1 / 3) * 2, 1))

        self.vdeconv1 = nn.ConvTranspose2d(N1, 1, kernel_size=(int(f1 / 3) + 1, t1))
        self.vdeconv2 = nn.ConvTranspose2d(N1, 1, kernel_size=(int(f1 / 3) + 1, t1))
        self.vdeconv3 = nn.ConvTranspose2d(N1, 1, kernel_size=(int(f1 / 3) + 1, t1))
        self.vdeconv4 = nn.ConvTranspose2d(N1, 1, kernel_size=(int(f1 / 3) + 1, t1))

    def forward(self, x):
        x = self.vconv(x)
        x, idx = self.maxpool(x)

        x = self.hconv(x)

        s1 = x.shape

        x = x.view(s1[0], -1)

        x = F.relu(self.fc0(x))

        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x))
        x4 = F.relu(self.fc4(x))

        x1 = x1.view(s1[0], s1[1], s1[2], s1[3])
        x2 = x2.view(s1[0], s1[1], s1[2], s1[3])
        x3 = x3.view(s1[0], s1[1], s1[2], s1[3])
        x4 = x4.view(s1[0], s1[1], s1[2], s1[3])

        x1 = self.hdeconv1(x1)
        x2 = self.hdeconv2(x2)
        x3 = self.hdeconv3(x3)
        x4 = self.hdeconv4(x4)

        x1 = self.maxunpool1(x1, idx)
        x2 = self.maxunpool2(x2, idx)
        x3 = self.maxunpool3(x3, idx)
        x4 = self.maxunpool4(x4, idx)

        x1 = self.vdeconv1(x1)
        x2 = self.vdeconv2(x2)
        x3 = self.vdeconv3(x3)
        x4 = self.vdeconv4(x4)

        return x1, x2, x3, x4
