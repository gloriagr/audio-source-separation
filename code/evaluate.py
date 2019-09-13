import mir_eval
import numpy as np
from scipy.io import wavfile
import librosa

####################### MODIFY ##############################
#### additional for loop to evaluate multiple songs #########
# increase step to sample less points and decrease time
sample_step = 10
bass_gt_path = '../Recovered_Songs_bigger5/bass/050.wav'
bass_rec_path = r'C:\Users\daisp\datasets\DSD100\Sources\Test\050 - Zeno - Signs\bass.wav'
vocal_gt_path = '../Recovered_Songs_bigger5/vocals/050.wav'
vocal_rec_path = r'C:\Users\daisp\datasets\DSD100\Sources\Test\050 - Zeno - Signs\vocals.wav'
drums_gt_path = '../Recovered_Songs_bigger5/drums/050.wav'
drums_rec_path = r'C:\Users\daisp\datasets\DSD100\Sources\Test\050 - Zeno - Signs\drums.wav'
others_gt_path = '../Recovered_Songs_bigger5/others/050.wav'
others_rec_path = r'C:\Users\daisp\datasets\DSD100\Sources\Test\050 - Zeno - Signs\other.wav'
############################################################

sample_rate, offset, duration = 44100, 30 * 0.3, 170 * 0.3
gt_instruments_paths = [bass_gt_path, drums_gt_path, others_gt_path, vocal_gt_path]
rec_instruments_paths = [bass_rec_path, drums_rec_path, others_rec_path, vocal_rec_path]
gt_tracks, rec_tracks = [], []

for gt_path, rec_path in zip(gt_instruments_paths, rec_instruments_paths):
    gt_tracks.append(librosa.load(gt_path, sr=sample_rate, offset=offset, duration=duration)[0])
    rec_tracks.append(librosa.load(rec_path, sr=sample_rate)[0])

for gt_track, rec_track in zip(gt_tracks, rec_tracks):
    # gt_track and rec_track are 2-element tuples whose first element contains the actual samples
    # and the second element is the sample rate
    gt_track = gt_track[0:rec_track.shape[0]:sample_step]
    gt_track = np.transpose(gt_track.reshape(len(gt_track), 1))

final_gt = np.concatenate(gt_tracks, axis=0)
print("final_gt shape = " + str(final_gt.shape))

for rec_track in rec_tracks:
    # rec_track is a 2-element tuple whose first element contains the actual samples
    # and the second element is the sample rate
    rec_track = rec_track[0:rec_track.shape[0]:sample_step]
    rec_track = np.transpose(rec_track.reshape(len(rec_track), 1))

final_rec = np.concatenate(rec_tracks, axis=0)
print("final_gt shape = " + str(final_rec.shape))

SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(final_rec, final_gt)

print(SDR)
print(SIR)
print(SAR)
print(perm)
