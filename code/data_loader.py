from torch.utils.data.dataset import Dataset
import torch
# from torchvision import transforms
# from skimage import io, transform
import os

class SourceSepTrain(Dataset):
    def __init__(self, path="../Train/Mixtures/", transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        mixture_path = '../Train/Mixtures/'
        bass_path = '../Train/Bass/'
        vocals_path = '../Train/Vocals/'
        drums_path = '../Train/Drums/'
        others_path = '../Train/Others/'
        mixture = torch.load(mixture_path + self.list[index])
        # phase = torch.load(mixture_path+self.list[index]+'_p')
        bass = torch.load(bass_path + self.list[index])
        vocals = torch.load(vocals_path + self.list[index])
        drums = torch.load(drums_path + self.list[index])
        others = torch.load(others_path + self.list[index])
        # print(mixture)
        if self.transforms is not None:
            mixture = self.transforms(mixture)
            bass = self.transforms(bass)
            vocals = self.transforms(vocals)
            drums = self.transforms(drums)
            others = self.transforms(others)

        self.cache[index] = (mixture, bass, vocals, drums,others)
        return (mixture, bass, vocals, drums, others)

    def __len__(self):
        return len(self.list)  # length of how much data you have


class SourceSepVal(Dataset):
    def __init__(self, path='../Validation/Mixtures/', transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        mixture_path = '../Validation/Mixtures/'
        bass_path = '../Validation/Bass/'
        vocals_path = '../Validation/Vocals/'
        drums_path = '../Validation/Drums/'
        others_path = '../Validation/Others/'

        mixture = torch.load(mixture_path + self.list[index])
        # phase_file = self.list[index].replace('_m', '_p')
        # phase_file = phase_file.replace('.pt', '.npy')
        # phase = np.load(phases_path+phase_file[0], allow_pickle=True)
        # phase = torch.load(phases_path + self.phase_list[index])
        bass = torch.load(bass_path + self.list[index], pickle_module=cPickle)
        vocals = torch.load(vocals_path + self.list[index])
        drums = torch.load(drums_path + self.list[index])
        others = torch.load(others_path + self.list[index])

        if self.transforms is not None:
            mixture = self.transforms(mixture)
            bass = self.transforms(bass)
            vocals = self.transforms(vocals)
            drums = self.transforms(drums)
            others = self.transforms(others)

        self.cache[index] = (mixture, bass, vocals, drums, others)
        return (mixture, bass, vocals, drums, others)

    def __len__(self):
        return len(self.list)


class SourceSepTest(Dataset):
    def __init__(self, path='../Test/Mixtures/', transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        mixture_path = '../Test/Mixtures/'
        bass_path = '../Test/Bass/'
        vocals_path = '../Test/Vocals/'
        drums_path = '../Test/Drums/'
        others_path = '../Test/Others/'
        phase_path = '../Test/Phases/'

        phase_file = self.list[index].replace('_m', '_p')
        phase_file = phase_file.replace('.pt', '.npy')
        mixture = torch.load(mixture_path + self.list[index])
        #phase = np.load(phase_path+phase_file)
        # bass = torch.load(bass_path + self.list[index])
        # vocals = torch.load(vocals_path + self.list[index])
        # drums = torch.load(drums_path + self.list[index])
        # others = torch.load(others_path + self.list[index])

        if self.transforms is not None:
            mixture = self.transforms(mixture)
            # bass = self.transforms(bass)
            # vocals = self.transforms(vocals)
            # drums = self.transforms(drums)
            # others = self.transforms(others)

        self.cache[index] = (mixture, phase_file, self.list[index])
        return (mixture, phase_file, self.list[index])

    def __len__(self):
        return len(self.list)
