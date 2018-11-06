import cv2
import process
import torch.utils.data as data
import os
import cv2
import dlib
import torch
import numpy as np


class FerSet(data.Dataset):
    def __init__(self, path=None, is_align=True):
        super(FerSet, self).__init__()
        assert path is not None
        self.path = path
        self.setList = os.listdir(path)
        self.setLength = len(self.setList)
        predictor = '../model/shape_predictor_68_face_landmarks.dat'
        detector = '../model/haarcascade_frontalface_default.xml'
        self.faceAlign = process.FaceAlign(predictor, detector)
        self.is_align = is_align

    def __getitem__(self, item):
        label = self.setList[item].split('_')[-1][0]
        imgPath = os.path.join(self.path, self.setList[item])
        data = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # alignment
        if self.is_align:
            bb = dlib.rectangle(0, 0, 48, 48)
            landmarks = self.faceAlign.predict(data, bb)
            data = self.faceAlign.align(data, landmarks)
        return torch.Tensor([data.astype(np.float32)]), eval(label)

    def __len__(self):
        return self.setLength

