import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from build_model_original import *
# to add dropout, change the SepConvNet in build_model_original
# to run with the max_pool layer, import instead:
#from build_model_original_1 import *

import os
from data_loader import *
from torch.optim.lr_scheduler import MultiStepLR

mean_var_path = "Processed/"
if not os.path.exists('Weights'):
    os.makedirs('Weights')

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# --------------------------
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    # property
    def avg(self):
        return self.sum / self.count


# ------------------------------

inp_size = [513, 52]
t1 = 1
f1 = 513
t2 = 12
f2 = 1
N1 = 50
N2 = 30
NN = 128
alpha = 0.001
beta = 0.01
beta_vocals = 0.03
batch_size = 30  # 5 / 128
num_epochs = 200


class MixedSquaredError(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MixedSquaredError, self).__init__()

    def forward(self, pred_bass, pred_vocals, pred_drums, pred_others, gt_bass, gt_vocals, gt_drums, gt_others):
        L_sq = torch.sum((pred_bass - gt_bass).pow(2)) + torch.sum((pred_vocals - gt_vocals).pow(2)) + torch.sum(
            (pred_drums - gt_drums).pow(2))
        # L_sq = L_Sq + torch.sum((pred_others - gt_others).pow(2))
        L_other = torch.sum((pred_bass - gt_others).pow(2)) + torch.sum((pred_drums - gt_others).pow(2))
        # + torch.sum((pred_vocals-gt_others).pow(2))
        L_othervocals = torch.sum((pred_vocals - gt_others).pow(2))
        L_diff = torch.sum((pred_bass - pred_vocals).pow(2)) + torch.sum((pred_bass - pred_drums).pow(2)) + torch.sum(
            (pred_vocals - pred_drums).pow(2))

        return (L_sq - alpha * L_diff - beta * L_other - beta_vocals * L_othervocals)


def TimeFreqMasking(bass, vocals, drums, others, cuda=0):
    den = torch.abs(bass) + torch.abs(vocals) + torch.abs(drums) + torch.abs(others)
    if (cuda):
        den = den + 10e-8 * torch.cuda.FloatTensor(bass.size()).normal_()
    else:
        den = den + 10e-8 * torch.FloatTensor(bass.size()).normal_()

    bass = torch.abs(bass) / den
    vocals = torch.abs(vocals) / den
    drums = torch.abs(drums) / den
    others = torch.abs(others) / den

    return bass, vocals, drums, others


def train():
    cuda = torch.cuda.is_available()
    print("cuda: ", cuda)
    net = SepConvNet(t1, f1, t2, f2, N1, N2, inp_size, NN)
    criterion = MixedSquaredError()  # try other losses
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)  # 1e-4 #weight decay 0
    train_set = SourceSepTrain(transforms=None)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = SourceSepVal(transforms=None)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    minimum = 1e9
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        net.train()
        train_loss = Average()

        for (inp, gt_bass, gt_vocals, gt_drums, gt_others) in train_loader:
            mean = torch.mean(inp)
            std = torch.std(inp)
            inp_n = (inp - mean) / std

            inp = Variable(inp)
            inp_n = Variable(inp_n)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            gt_others = Variable(gt_others)
            if cuda:
                inp = inp.cuda()
                inp_n = inp_n.cuda()
                gt_bass = gt_bass.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()
                gt_others = gt_others.cuda()
            optimizer.zero_grad()
            o_bass, o_vocals, o_drums, o_others = net(inp_n)

            mask_bass, mask_vocals, mask_drums, mask_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others, cuda)
            pred_drums = inp * mask_drums
            pred_vocals = inp * mask_vocals
            pred_bass = inp * mask_bass
            pred_others = inp * mask_others

            loss = criterion(pred_bass, pred_vocals, pred_drums, pred_others, gt_bass, gt_vocals, gt_drums, gt_others)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), inp.size(0))

        val_loss = Average()
        net.eval()
        for i, (val_inp, gt_bass, gt_vocals, gt_drums, gt_others) in enumerate(val_loader):
            val_mean = torch.mean(val_inp)
            val_std = torch.std(val_inp)
            val_inp_n = (val_inp - val_mean) / val_std

            val_inp = Variable(val_inp)
            val_inp_n = Variable(val_inp_n)
            gt_bass = Variable(gt_bass)
            gt_vocals = Variable(gt_vocals)
            gt_drums = Variable(gt_drums)
            gt_others = Variable(gt_others)
            if cuda:
                val_inp = val_inp.cuda()
                val_inp_n = val_inp_n.cuda()
                gt_bass = gt_bass.cuda()
                gt_vocals = gt_vocals.cuda()
                gt_drums = gt_drums.cuda()
                gt_others = gt_others.cuda()

            o_bass, o_vocals, o_drums, o_others = net(val_inp_n)
            mask_bass, mask_vocals, mask_drums, mask_others = TimeFreqMasking(o_bass, o_vocals, o_drums, o_others, cuda)
            pred_drums = val_inp * mask_drums
            pred_vocals = val_inp * mask_vocals
            pred_bass = val_inp * mask_bass
            pred_others = val_inp * mask_others

            vloss = criterion(pred_bass, pred_vocals, pred_drums, pred_others, gt_bass, gt_vocals, gt_drums, gt_others)
            val_loss.update(vloss.item(), val_inp.size(0))

        scheduler.step()
        print("Epoch {}, Training Loss: {}, Validation Loss: {}".format(epoch + 1, train_loss.avg(), val_loss.avg()))
        torch.save(net.state_dict(), 'Weights/Weights_{}_{}.pth'.format(epoch + 1, val_loss.avg()))
        if val_loss.avg() < minimum:
            epochs_no_improve = 0
            minimum = val_loss.avg()
        else:
            epochs_no_improve += 1

        if epochs_no_improve > 7:
            break

    return net


def test(model):
    model.eval()


if __name__ == "__main__":
    train()
