import os
from shutil import copyfile
from natsort import natsorted

folders = ["Bass", "Drums", "Mixtures", "Others", "Phases", "Vocals"]

src_path_train_1 = "../Processed/"
src_path_train_2 = "../Val/"
dst_path_train = "../Train/"
if not os.path.exists(dst_path_train):
    os.makedirs(dst_path_train)
for folder in folders:
    if not os.path.exists(dst_path_train + folder):
        os.makedirs(dst_path_train + folder)

src_path_val = "../Processed/"
dst_path_val = "../Validation/"
if not os.path.exists(dst_path_val):
    os.makedirs(dst_path_val)
for folder in folders:
    if not os.path.exists(dst_path_val + folder):
        os.makedirs(dst_path_val + folder)

src_path_test = "../Processed/"
dst_path_test = "../Test/"
if not os.path.exists(dst_path_test):
    os.makedirs(dst_path_test)
for folder in folders:
    if not os.path.exists(dst_path_test + folder):
        os.makedirs(dst_path_test + folder)

#import pdb;pdb.set_trace()
for root, dirs, f in os.walk(src_path_train_1):
    for direc in natsorted(dirs):
        for root_, dir, files in os.walk(src_path_train_1+ direc):
            for i in files:
                if int(i[:3]) <= 80:
                    copyfile(os.path.join(src_path_train_1, direc, i), os.path.join(dst_path_train, direc, i))

for root, dirs, f in os.walk(src_path_train_2):
    for direc in natsorted(dirs):
        for root_, dir, files in os.walk(src_path_train_2 + direc):
            for i in files:
                copyfile(os.path.join(src_path_train_2, direc, i), os.path.join(dst_path_train, direc, i))


for root, dirs, f in os.walk(src_path_val):
    for direc in natsorted(dirs):
        for root_, dir, files in os.walk(src_path_val + direc):
            for i in files:
                if int(i[:3]) > 80 and int(i[:3]) <= 90:
                    copyfile(os.path.join(src_path_val, direc, i), os.path.join(dst_path_val, direc, i))

for root, dirs, f in os.walk(src_path_test):
    for direc in natsorted(dirs):
        for root_, dir, files in os.walk(src_path_test + direc):
            for i in files:
                if int(i[:3]) > 90:
                    copyfile(os.path.join(src_path_test, direc, i), os.path.join(dst_path_test, direc, i))
