import mir_eval
import numpy as np
from scipy.io import wavfile
import librosa
import os
from shutil import copyfile
from natsort import natsorted

# increase step to sample less points and decrease time
sample_step = 10
Rec_path = '../Recovered_Songs_bigger5'

# location of the DSD100/Sources/Dev folder
Gt_path = '../../adversarial-audio-separation/AdversarialAudioSeparation/Data/DSD100/DSD100/Sources/Dev'
# or if we want the original 50-50 divide, then the location of the DSD100/Sources/Test folder
#Gt_path = '../../adversarial-audio-separation/AdversarialAudioSeparation/Data/DSD100/DSD100/Sources/Test'

if not os.path.exists('Recovered_Songs_Bigger_6'):
    os.makedirs('Recovered_Songs_Bigger_6')

bass_rec_path = ''
bass_gt_path = ''
vocals_rec_path = ''
vocals_gt_path = ''
drums_rec_path = ''
drums_gt_path = ''
others_rec_path = ''
others_gt_path = ''
############################################################
for root, dirs, files in os.walk(Rec_path):
    for dir in dirs:
        for _, _, files_ in os.walk(os.path.join(Rec_path, dir)):
            for file in files_:
                if not os.path.exists(os.path.join('Recovered_Songs_Bigger_6/', file[:3])):
                    os.makedirs(os.path.join('Recovered_Songs_Bigger_6/', file[:3]))
        for _, _, files_ in os.walk(os.path.join(Rec_path, dir)):
            for file in files_:
                copyfile(os.path.join(Rec_path, dir, file), os.path.join('Recovered_Songs_Bigger_6/', file[:3], dir + '.wav'))

Rec_path = 'Recovered_Songs_Bigger_6'

SDR_list = []
SAR_list = []
SIR_list = []

# if we want the original 50-50 divide, uncomment this instead:
#for aa, bb in zip(natsorted(os.listdir(Rec_path)), natsorted(os.listdir(Gt_path))):
for aa, bb in zip(natsorted(os.listdir(Rec_path)), natsorted(os.listdir(Gt_path))[-10:]):
    print(aa, bb) # make sure they match, and change the above line accordingly
    list_1 = natsorted(os.listdir(os.path.join(Rec_path, aa)))
    list_2 = natsorted(os.listdir(os.path.join(Gt_path, bb)))
    for file_rec, file_gt in zip(list_1, list_2):
        if file_rec[0:-4] == 'bass':
            bass_gt_path = os.path.join(Gt_path, bb, file_gt)
            bass_rec_path = os.path.join(Rec_path, aa, file_rec)
        elif file_rec[0:-4] == 'drums':
            drums_gt_path = os.path.join(Gt_path, bb, file_gt)
            drums_rec_path = os.path.join(Rec_path, aa, file_rec)
        elif file_rec[0:-4] == 'vocals':
            vocals_gt_path = os.path.join(Gt_path, bb, file_gt)
            vocals_rec_path = os.path.join(Rec_path, aa, file_rec)
        elif file_rec[0:-4] == 'others':
            others_gt_path = os.path.join(Gt_path, bb, file_gt)
            others_rec_path = os.path.join(Rec_path, aa, file_rec)

    sample_rate, offset, duration = 44100, 30 * 0.3, 170 * 0.3
    gt_instruments_paths = [bass_gt_path, drums_gt_path, others_gt_path, vocals_gt_path]
    rec_instruments_paths = [bass_rec_path, drums_rec_path, others_rec_path, vocals_rec_path]
    gt_tracks, rec_tracks = [], []

    # skip = False
    # i = 0

    for gt_path, rec_path in zip(gt_instruments_paths, rec_instruments_paths):
       gt_tracks.append(librosa.load(gt_path, sr=sample_rate, offset=offset, duration=duration)[0])
       # if np.count_nonzero(gt_tracks[i]) == 0:
       #     skip = True
       #     break
       # i += 1
       rec_tracks.append(librosa.load(rec_path, sr=sample_rate)[0])

    # if skip:
    #     continue

    # gt_track and rec_track are 2-element tuples whose first element contains the actual samples
    # and the second element is the sample rate
    length = rec_tracks[0].shape[0]

    gt_tracks[:] = [gt_track[0:length:sample_step] for gt_track in gt_tracks]
    len_gt = gt_tracks[0].shape[0]
    gt_tracks[:] = [np.transpose(gt_track.reshape(len_gt, 1)) for gt_track in gt_tracks]
    final_gt = np.concatenate(gt_tracks, axis=0)

    rec_tracks[:] = [rec_track[0:length:sample_step]for rec_track in rec_tracks]
    len_rec = rec_tracks[0].shape[0]
    rec_tracks[:] = [np.transpose(rec_track.reshape(len_rec, 1)) for rec_track in rec_tracks]
    final_rec = np.concatenate(rec_tracks, axis=0)

    SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(final_gt, final_rec)
    SDR_list.append(SDR)
    SIR_list.append(SIR)
    SAR_list.append(SAR)


SDR = (np.stack(SDR_list)).T
SIR = (np.stack(SIR_list)).T
SAR = (np.stack(SAR_list)).T

print("SDR")
print(SDR[0], SDR[1], SDR[2], SDR[3])
print("SIR")
print(SIR[0], SIR[1], SIR[2], SIR[3])
print("SAR")
print(SAR[0], SAR[1], SAR[2], SAR[3])

prints = ["SDR: ", "SIR: ", "SAR: "]
prints2 = ['bass: ', 'drums: ', 'others: ', 'vocals: ']
vals = [SDR_list, SIR_list, SAR_list]
for j in range(3):
    print(prints[j])
    for i in range(4):
        list_ = [s[i] for s in vals[j]]
        mean = np.mean(list_)
        std = np.std(list_)
        print(prints2[i], mean, "+-", std)

