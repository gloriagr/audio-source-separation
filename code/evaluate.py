import mir_eval
import numpy as np
from scipy.io import wavfile
import librosa
import os

####################### MODIFY ##############################
#### additional for loop to evaluate multiple songs #########
# increase step to sample less points and decrease time
dataset_path = 'C:/Users/daisp/datasets/DSD100/'
sample_step = 10
bass_rec_path = '../Recovered_Songs_bigger5/bass/001.wav'
bass_gt_path = dataset_path + 'Sources/Test/001 - ANiMAL - Clinic A/bass.wav'
vocal_rec_path = '../Recovered_Songs_bigger5/vocals/001.wav'
vocal_gt_path = dataset_path + 'Sources/Test/001 - ANiMAL - Clinic A/vocals.wav'
drums_rec_path = '../Recovered_Songs_bigger5/drums/001.wav'
drums_gt_path = dataset_path + 'Sources/Test/001 - ANiMAL - Clinic A/drums.wav'
others_rec_path = '../Recovered_Songs_bigger5/others/001.wav'
others_gt_path = dataset_path + 'Sources/Test/001 - ANiMAL - Clinic A/other.wav'
############################################################

sample_rate, offset, duration = 44100, 30 * 0.3, 170 * 0.3
gt_instruments_paths = [bass_gt_path, drums_gt_path, others_gt_path, vocal_gt_path]
rec_instruments_paths = [bass_rec_path, drums_rec_path, others_rec_path, vocal_rec_path]
gt_tracks, rec_tracks = [], []

for gt_path, rec_path in zip(gt_instruments_paths, rec_instruments_paths):
    gt_tracks.append(librosa.load(gt_path, sr=sample_rate, offset=offset, duration=duration)[0])
    rec_tracks.append(librosa.load(rec_path, sr=sample_rate)[0])


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

print(SDR)
print(SIR)
print(SAR)
print(perm)
