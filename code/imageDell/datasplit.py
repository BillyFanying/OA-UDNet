import os
import random

import cv2
from tqdm import tqdm

target_train_Dir = r'E:\FY_Data\YSL_Code\newData_othermodel\trainAndVal\256_elems'
target_test_Dir = r'E:\FY_Data\YSL_Code\newData_othermodel\test\256_elems'
original_dir = r'E:\FY_Data\Data_Othermodel\256_elems'

if not os.path.exists(target_train_Dir):
    os.mkdir(target_train_Dir)
if not os.path.exists(target_test_Dir):
    os.mkdir(target_test_Dir)
buwei_names = os.listdir(original_dir)
epoch = 0
for beiwei in buwei_names:

    buwei_path = os.path.join(original_dir, beiwei)
    img_names = os.listdir(buwei_path)
    img_len = len(img_names)
    trainVal_size = int(0.9 * img_len)
    test_size = img_len - trainVal_size
    train_img_names = img_names[:trainVal_size]
    test_img_names = img_names[trainVal_size:]

    progress_bar = tqdm(train_img_names, desc=f"Epoch {epoch + 1}/{len(buwei_names)}", total=trainVal_size)
    for tname in train_img_names:

        img_path = os.path.join(buwei_path, tname)
        save_path = os.path.join(target_train_Dir, tname)
        img = cv2.imread(img_path)
        cv2.imwrite(save_path, img)
        progress_bar.update(1)
    progress_bar = tqdm(test_img_names, desc=f"Epoch {epoch + 1}/{len(buwei_names)}", total=test_size)
    for tname in test_img_names:
        img_path = os.path.join(buwei_path, tname)
        save_path = os.path.join(target_test_Dir, tname)
        img = cv2.imread(img_path)
        cv2.imwrite(save_path, img)
        progress_bar.update(1)
    epoch = epoch + 1


print('over')





