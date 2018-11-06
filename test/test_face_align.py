# coding = utf-8
import os
import sys
sys.path.append('./../')
from process.face_align import *
import glob
import cv2
import dlib
import numpy as np
# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path = '../data/fer2013/train/'
model1 = '../model/shape_predictor_68_face_landmarks.dat' # 68 points in people face
# model = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
model = '../model/haarcascade_frontalface_default.xml' # haar feature
if __name__ == "__main__":
    whole_set = glob.glob(data_path + "*.bmp")
    # whole_set = ['../data/test']
    align = FaceAlign(model1, model) # a class, to use it's function
    faceCascade = cv2.CascadeClassifier(model)
    cnt = 0
    ex_length = len(data_path)
    # print(a)
    for img_path in whole_set:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_path = img_path[ex_length:-4]
        # print(img_path)
        bb = align.detect(img)
        if bb == []:
            bb = [dlib.rectangle(0, 0, 48, 48)]
        # print(bb)
        landmarks = align.predict(img, bb[0])
        # landmarks = align.predict(img, bb)
        for landmark in landmarks:
            cv2.circle(img, (landmark[0], landmark[1]), 0, color=(0, 0, 255), thickness=2)
        r = align.align(img, landmarks)

        # cv2.imshow('align', r)
        d = np.array(r)#.reshape(1, -1)
        # print('d = ',d)
        np.savetxt('./../data/fer2013/train_aligned/'+img_path+'.txt', d, fmt = '%d')
        # print (d.shape)
        cnt = cnt + 1
        # cv2.imshow('face', img)
        # k = cv2.waitKey(0)
        # if k == ord("q"):
        #     break
    print('convert finished')