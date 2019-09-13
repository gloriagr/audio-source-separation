import librosa
import numpy as np
#import mathplotlib.pyplot as plt
import pickle
import torch
import os
import re

#change to the path where DSD100 is in
path= "../../adversarial-audio-separation/AdversarialAudioSeparation/Data/DSD100/DSD100/"
path_mixtures = path + "Mixtures/Dev/"
path_sources = path + "Sources/Dev/"
mean_var_path= "../Processed/"
destination_path = "../Processed/Mixtures"
phase_path= "../Processed/Phases"
bass_path="../Processed/Bass"
vocals_path="../Processed/Vocals"
drums_path="../Processed/Drums"
others_path="../Processed/Others"
#source_dest_paths=[vocals_path,bass_path,drums_path,others_path]
source_dest_paths=[bass_path,drums_path,others_path, vocals_path]


path_val_mixtures = path + "Mixtures/Test/"
path_val_sources = path + "Sources/Test/"
validation_path = "../Val/Mixtures"
val_phase_path= "../Val/Phases"
val_bass_path="../Val/Bass"
val_vocals_path="../Val/Vocals"
val_drums_path="../Val/Drums"
val_others_path="../Val/Others"
#source_val_paths=[val_vocals_path,val_bass_path,val_drums_path,val_others_path]
source_val_paths=[val_bass_path,val_drums_path,val_others_path, val_vocals_path]


### test paths for now same as the validation path!
path_test_mixtures = path + "Mixtures/Test/"
path_test_sources = path + "Sources/Test/"
testing_path = "../Test/Mixtures"
test_phase_path= "../Test/Phases"
test_bass_path="../Test/Bass"
test_vocals_path="../Test/Vocals"
test_drums_path="../Test/Drums"
test_others_path="../Test/Others"
#source_test_paths=[test_vocals_path,test_bass_path,test_drums_path,test_others_path]
source_test_path=[test_bass_path, test_drums_path, test_others_path, test_vocals_path]


def process(file_path, direc, destination_path, phase_bool, destination_phase_path):
    signal, samp_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(signal, samp_rate)
    regex = re.compile(r'\d+')
    index = regex.findall(direc)
    for start in range(30, int(200)):
        wave_array, fs = librosa.load(file_path, sr=44100, offset=start*0.3, duration=0.3)

        mag, phase = librosa.magphase(librosa.stft(wave_array, n_fft=1024, hop_length=256, window='hann', center='True'))
        #mean+=mag
        #num_segments+=1;
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # magnitude stored as tensor, phase as np array
        save_path = os.path.join(destination_path, (index[0] + "_" + str(start) + '_m.pt'))
        torch.save(torch.from_numpy(np.expand_dims(mag, axis=0)), save_path)

        if phase_bool:
            if not os.path.exists(destination_phase_path):
                os.makedirs(destination_phase_path)
            np.save(os.path.join(destination_phase_path, (index[0] + "_" +str(start) + '_p.npy')), phase)



#--------- training data-------------------------------------

import pdb;pdb.set_trace()
for curr_path, dirs, files in os.walk(path_mixtures):
  for direc in dirs:
      print('working with training '+ direc)
      for c,d,f in os.walk(path_mixtures + direc):
          process(os.path.join(path_mixtures, direc, f[0]), direc, destination_path, True, phase_path)


for curr_path, dirs, files in os.walk(path_sources):
    for direc in dirs:
        print('source with training '+ direc)
        for c,d,files in os.walk(path_sources + direc):
            for i, f in enumerate(sorted(files)):
                process(os.path.join(path_sources, direc, f), direc, source_dest_paths[i], False, phase_path)



#------------------------ Validation data-----------------------------------

for curr_path, dirs, files in os.walk(path_val_mixtures):
    for direc in dirs:
        print('working with validation '+ direc)
        for c,d,f in os.walk(path_val_mixtures + direc):
            process(os.path.join(path_val_mixtures, direc, f[0]), direc,validation_path, True, val_phase_path)

for curr_path, dirs, files in os.walk(path_val_sources):
    for direc in dirs:
        print('source with validation '+ direc)
        for c,d,files in os.walk(path_val_sources + direc):
           # for i in range(0,4):
           #     print(file[i])
           for i, f in enumerate(sorted(files)):
                process(os.path.join(path_val_sources, direc, f), direc, source_val_paths[i], False, val_phase_path)

#----------------------Testing data-------------------------------------------

for curr_path, dirs, files in os.walk(path_test_mixtures):
    for direc in dirs:
        print('working with validation '+ direc)
        for c,d,f in os.walk(path_test_mixtures + direc):
            process(os.path.join(path_test_mixtures,direc,f[0]),direc,testing_path,True,test_phase_path)

for curr_path, dirs, files in os.walk(path_test_sources):
    for direc in dirs:
        print('source with testset '+ direc)
        for c,d,files in os.walk(path_test_sources + direc):
          #  for i in range(0,4):
          #      print(file[i])
          for i, f in enumerate(sorted(files)):
                process(os.path.join(path_test_sources,direc,f),direc,source_test_path[i],False,test_phase_path)
