# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import (LOGGER, Timeout, check_requirements, clip_coords, increment_path, is_ascii, is_chinese,
                           try_except, user_config_dir, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness
import numpy as np


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box):
        # Grey For Human Detection Box
        color = (64, 64, 64)

        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

    def mergeMarks(self, enterMarks, exitMarks):
        for SetMarks in enterMarks:
            self.im = cv2.line(self.im, (SetMarks[0][0], SetMarks[0][1]), (SetMarks[1][0], SetMarks[1][1]),
                               color=(42, 181, 26), thickness=8)
        for SetMarks in exitMarks:
            self.im = cv2.line(self.im, (SetMarks[0][0], SetMarks[0][1]), (SetMarks[1][0], SetMarks[1][1]),
                               color=(16, 23, 233), thickness=8)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def compareImage(img1, img2):
    image1 = imagehash.average_hash(Image.open(img1))
    image2 = imagehash.average_hash(Image.open(img2))

    threshold = 15
    if image1 - image2 < threshold:
        # Images are similar
        return True
    else:
        # images are not similar
        return False


def enterCheck(xyxy, im):
    # Take in Original Image
    crop = cropDetected(xyxy, im)

    result = False

    folderName = "runs/entered"
    filesEntered = os.listdir(folderName)
    if len(filesEntered) != 0:
        for fileE in filesEntered:
            # People are not the same = New Person Entered += 1
            if not compareImage(crop, fileE):
                try:
                    now = datetime.now().strftime("%H%M%S")
                    fileName = folderName + "/" + str(now) + "enterDetected.jpeg"
                    cv2.imwrite(fileName, crop)
                    result = True

                    folderName2 = "runs/exited"
                    filesExited = os.listdir(folderName2)

                    if (filesExited) != 0:
                        for fileX in filesExited:
                            # People are  the same = Means that person has been exited before therefore delete exited person should happen
                            # So if that person exit again, he can be counted as exited when he reached the exit
                            if compareImage(crop, fileX):
                                try:
                                    os.remove(folderName2 + "/" + fileX)
                                except:
                                    print("Delete people from Exit Folder Failed")
                except:
                    print("Add New Person entered failed")
                print("New Person Entered")
    else:
        try:
            now = datetime.now().strftime("%H%M%S")
            fileName = folderName + "/" + str(now) + "enterDetected.jpeg"
            cv2.imwrite(fileName, crop)
            result = True

            folderName2 = "runs/exited"
            filesExited = os.listdir(folderName2)

            if (filesExited) != 0:
                for fileX in filesExited:
                    # People are  the same = Means that person has been exited before therefore delete exited person should happen
                    # So if that person exit again, he can be counted as exited when he reached the exit
                    if compareImage(crop, fileX):
                        try:
                            os.remove(folderName2 + "/" + fileX)
                        except:
                            print("Delete people from Exit Folder Failed")
        except:
            print("Add New Person entered failed")
        print("New Person Entered")

    return result


def exitCheck(xyxy, im):
    # Take in Original Image
    crop = cropDetected(xyxy, im)

    result = False

    folderName = "runs/exited"
    filesExited = os.listdir(folderName)

    if len(filesExited) != 0:
        for fileX in filesExited:
            # People are not the same = New Person Exited -= 1
            if not compareImage(crop, fileX):
                try:
                    now = datetime.now().strftime("%H%M%S")
                    fileName = folderName + "/" + str(now) + "exitDetected.jpeg"
                    cv2.imwrite(fileName, crop)
                    result = True

                    folderName2 = "runs/entered"
                    filesEntered = os.listdir(folderName2)

                    if (filesEntered) != 0:
                        for fileE in filesEntered:
                            # People are  the same = Means that person has entered before therefore delete this entered person
                            # So if that person enters again, he will be counted as entered
                            if compareImage(crop, fileE):
                                try:
                                    os.remove(folderName2 + "/" + fileE)
                                except:
                                    print("Delete people from Exit Folder Failed")
                except:
                    print("Add New Person entered failed")
                print("New Person Entered")
    else:
        try:
            now = datetime.now().strftime("%H%M%S")
            fileName = folderName + "/" + str(now) + "exitDetected.jpeg"
            cv2.imwrite(fileName, crop)
            result = True

            folderName2 = "runs/entered"
            filesEntered = os.listdir(folderName2)

            if (filesEntered) != 0:
                for fileE in filesEntered:
                    # People are  the same = Means that person has entered before therefore delete this entered person
                    # So if that person enters again, he will be counted as entered
                    if compareImage(crop, fileE):
                        try:
                            os.remove(folderName2 + "/" + fileE)
                        except:
                            print("Delete people from Exit Folder Failed")
        except:
            print("Add New Person entered failed")
        print("New Person Entered")

    return result

def detectEnter(xyxy, im):
    # Take in Marked Image
    crop = cropDetected(xyxy, im)

    # Code Adapted from Kinght 金, 2017.
    lowerHSVEnter = (62, 215, 180)
    upperHSVEnter = (64, 219, 182)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, lowerHSVEnter, upperHSVEnter)
    count = np.sum(np.nonzero(thresh))
    # End of Code Adapted

    # Count == 0 means no green detected
    return count


def detectExit(xyxy, im):
    # Take in Marked Image
    crop = cropDetected(xyxy, im)

    lowerHSVExit = (0, 238, 232)
    upperHSVExit = (1, 238, 233)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, lowerHSVExit, upperHSVExit)
    count = np.sum(np.nonzero(thresh))

    # Count == 0 means no red detected
    return count


def cropDetected(xyxy, im, gain=1.02, pad=10, square=False):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::1]
    return crop


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)
