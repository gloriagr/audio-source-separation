import torch
import numpy as np
import glob
import re
import os
from build_model_original import SepConvNet
# to add dropout, change the SepConvNet
# to run with the max_pool layer, import instead:
#from build_model_original_1 import SepConvNet

from torch.utils.data import DataLoader
from data_loader import SourceSepTest
from post_processing import reconstruct
from train_model import TimeFreqMasking
from tqdm import tqdm


if __name__ == '__main__':
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
    batch_size = 1
    num_epochs = 100

    destination_path = '../AudioResults/'
    # location of the phases file created in pre-processing
    phase_path = '../Test/Phases/'
    # if we want the original 50-50 divide, comment the line above,and uncomment instead:
    #phase_path = '../Val/Phases/'
    vocals_directory = '../AudioResults/vocals'
    drums_directory = '../AudioResults/drums'
    bass_directory = '../AudioResults/bass'
    others_directory = '../AudioResults/others'

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    if not os.path.exists(bass_directory):
        os.makedirs(bass_directory)
    if not os.path.exists(drums_directory):
        os.makedirs(drums_directory)
    if not os.path.exists(others_directory):
        os.makedirs(others_directory)
    if not os.path.exists(vocals_directory):
        os.makedirs(vocals_directory)
    net = SepConvNet(t1, f1, t2, f2, N1, N2, inp_size, NN)
    # put the chosen weights file
    net.load_state_dict(torch.load('Weights/Weights_14_0.020168437798890996.pth'))
    # if the script runs without gpu, uncomment the next line instead:
    #net.load_state_dict(torch.load('Weights/Weights_14_0.020168437798890996.pth',  map_location='cpu'))
    net.eval()
    test_set = SourceSepTest(transforms=None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    for (test_inp, test_phase_file, file_str) in test_loader:

        test_phase = np.load(phase_path + test_phase_file[0])

        # if int(index[0]) == 8 or int(index[0]) == 6 or int(index[0]) == 22:
        #   continue

        mean = torch.mean(test_inp)
        std = torch.std(test_inp)
        test_inp_n = (test_inp - mean) / std

        bass_mag, vocals_mag, drums_mag, others_mag = net(test_inp_n)
        bass_mag, vocals_mag, drums_mag, others_mag = TimeFreqMasking(bass_mag, vocals_mag, drums_mag, others_mag)
        bass_mag = bass_mag * test_inp
        vocals_mag = vocals_mag * test_inp
        drums_mag = drums_mag * test_inp
        others_mag = others_mag * test_inp

        regex = re.compile(r'\d+')
        index = regex.findall(file_str[0])
        #reconstruct(test_phase, bass_mag, vocals_mag, drums_mag, others_mag, index[0], index[1], destination_path)
        reconstruct(test_phase, vocals_mag, index[0], index[1], destination_path)
