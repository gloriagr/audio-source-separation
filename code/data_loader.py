from torch.utils.data.dataset import Dataset
import torch
# from torchvision import transforms
# from skimage import io, transform
import os

''' train_mixture = "Train/Mixtures/"
train_bass = 'Train/Bass/'
train_vocals = 'Train/Vocals/'
train_drums = 'Train/Drums/'
train_others = 'Train/Others/'

val_mixture = "Validation/Mixtures/"
val_bass = 'Validation/Bass/'
val_vocals = 'Validation/Vocals/'
val_drums = 'Validation/Drums/'
val_others = 'Validation/Others/'

test_mixture = "Test/Mixtures/"
test_phases = 'Test/Phases/' '''

train_mixture = "Processed/Mixtures/"
train_bass = 'Processed/Bass/'
train_vocals = 'Processed/Vocals/'
train_drums = 'Processed/Drums/'
train_others = 'Processed/Others/'

val_mixture = "Val/Mixtures/"
val_bass = 'Val/Bass/'
val_vocals = 'Val/Vocals/'
val_drums = 'Val/Drums/'
val_others = 'Val/Others/'

test_mixture = "Val/Mixtures/"
test_phases = 'Val/Phases/'


class SourceSepTrain(Dataset):
    def __init__(self, path=train_mixture, transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        mixture_path = train_mixture
        bass_path = train_bass
        vocals_path = train_vocals
        drums_path = train_drums
        others_path = train_others
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

        self.cache[index] = (mixture, bass, vocals, drums, others)
        return (mixture, bass, vocals, drums, others)

    def __len__(self):
        return len(self.list)  # length of how much data you have


class SourceSepVal(Dataset):
    def __init__(self, path=val_mixture, transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        # stuff
        mixture_path = val_mixture
        bass_path = val_bass
        vocals_path = val_vocals
        drums_path = val_drums
        others_path = val_others

        mixture = torch.load(mixture_path + self.list[index])
        # phase_file = self.list[index].replace('_m', '_p')
        # phase_file = phase_file.replace('.pt', '.npy')
        # phase = np.load(phases_path+phase_file[0], allow_pickle=True)
        # phase = torch.load(phases_path + self.phase_list[index])
        bass = torch.load(bass_path + self.list[index])
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
    def __init__(self, path=test_mixture, transforms=None):
        # assuming this to be the directory containing all the magnitude spectrum
        # for all songs and all segments used in training
        self.path = path
        self.list = os.listdir(self.path)
        self.transforms = transforms
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        mixture_path = test_mixture
        # bass_path = 'Val/Bass/'
        # vocals_path = '../Val/Vocals/'
        # drums_path = '../Val/Drums/'
        # others_path = '../Val/Others/'
        phase_path = test_phases

        phase_file = self.list[index].replace('_m', '_p')
        phase_file = phase_file.replace('.pt', '.npy')
        mixture = torch.load(mixture_path + self.list[index])
        # phase = np.load(phase_path+phase_file)
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
