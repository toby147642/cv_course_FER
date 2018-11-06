# -*-coding:utf-8-*-
import numpy as np
import pickle
import random
import os
import cv2
import glob
import dlib
from face_align import FaceAlign



def dataAugmentation(set_name=None):
    assert set_name is not None
    print set_name
    dim = 48
    train_label = None
    data = []
    with open('../data/fer2013/{}'.format(set_name), 'rb') as f:
        samples = pickle.load(f)
        for i, d in enumerate(samples['data']):
            # 随机采样，平移
            d = np.reshape(d, (dim, dim))
            raw = np.array([d], dtype=np.uint8)
            # 原始数据
            data.append(raw)
            padding1 = np.zeros((2, dim), dtype=np.uint8)
            padding2 = np.zeros((dim+4, 2), dtype=np.uint8)

            temp_d = np.concatenate((d, padding1), axis=0)
            temp_d = np.concatenate((padding1, temp_d), axis=0)
            temp_d = np.concatenate((temp_d, padding2), axis=1)
            temp_d = np.concatenate((padding2, temp_d), axis=1)

            for j in range(4):
                rd_x = random.randint(dim/2, dim/2+4)
                rd_y = random.randint(dim/2, dim/2+4)
                if rd_x == dim/2+2 and rd_y == dim/2+2:
                    rd_x = random.randint(dim/2, dim/2+4)
                    rd_y = random.randint(dim/2, dim/2+4)
                r = temp_d[rd_y - dim/2:rd_y + dim/2, rd_x - dim/2:rd_x + dim/2]

                data.append(np.array([r], dtype=np.uint8))
            # 映射
            rotation1 = raw[:, ::-1, :]
            rotation2 = raw[:, :, ::-1]
            data.append(rotation1)
            data.append(rotation2)
            # 尺度
            up_scale = 1.1
            up_sample = np.zeros((1, dim, dim), dtype=np.uint8)
            down_sample = np.zeros((1, dim, dim), dtype=np.uint8)
            down_scale = 0.9
            temp = cv2.resize(d,
                              (0, 0),
                              fx=up_scale,
                              fy=up_scale,
                              interpolation=cv2.INTER_CUBIC)
            up_sample[:,:,:] = temp[
                        temp.shape[1]/2-dim/2:temp.shape[1]/2+dim/2,
                        temp.shape[0]/2-dim/2:temp.shape[0]/2+dim/2]

            temp = cv2.resize(d, (0, 0), fx=down_scale, fy=down_scale)
            down_sample[
                :,
            dim/2-(temp.shape[1]+1)/2:dim/2+temp.shape[1]/2,
            dim/2-(temp.shape[0]+1)/2:dim/2+temp.shape[0]/2] = temp[:,:]
            data.append(up_sample)
            data.append(down_sample)
            # 加椒盐噪声
            noise = raw.copy()
            for k in range(25):
                rx = random.randint(0, dim-1)
                ry = random.randint(0, dim-1)
                noise[0, rx, ry] = 255
            for k in range(25):
                rx = random.randint(0, dim-1)
                ry = random.randint(0, dim-1)
                noise[0, rx, ry] = 255
            data.append(noise)
            if (i+1) % 5000 == 0:
                print "data augmentation on : {}".format(i+1)
            # r = np.array([r])
            # I = np.concatenate((raw[0,:,:], r[0,:,:], rotation1[0,:,:], rotation2[0,:,:],up_sample[0,:,:],down_sample[0,:,:],noise[0,:,:]), axis=1)
            # cv2.imshow('w', I)
            # k = cv2.waitKey()
            # if k == ord('q'):
            #     break

        data = np.array(data)
        train_label = np.array(samples['label']).repeat(10)

    print data.shape
    print train_label.shape

    return data, train_label


def data_set_load(set_name=None, is_validate=False):
    assert set_name is not None
    dim = 48
    vlabels = None

    with open('../data/fer2013/{}'.format(set_name), 'rb') as f:
        samples = pickle.load(f)
        data = []
        val = []
        for i, d in enumerate(samples['data']):
            # 随机采样，平移
            d = np.reshape(d, (dim, dim))
            raw = np.array([d], dtype=np.uint8)
            if i >= 4500 and is_validate:
                val.append(raw)
                continue
            # 原始数据
            data.append(raw)
        data = np.array(data)
        if is_validate:
            labels = np.array(samples['label'][:4500])
            val = np.array(val)
            vlabels = np.array(samples['label'][4500:])
        else:
            labels = np.array(samples['label'])
    print data.shape
    print val.shape
    return data, labels, val, vlabels


def data_set_refine():
    data_path = '../data/fer2013/test'
    name_list = os.listdir(data_path)
    length = 48*48
    all_data = None
    all_label = []
    print len(name_list)
    for i, name in enumerate(name_list):
        if (i+1) % 30 == 0:
            print "iterations:{}".format(i+1)
        label = eval(name.split('_')[-1][0])
        imgPath = os.path.join(data_path, name)
        data = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        data = np.reshape(data, (1,length)).astype(np.uint8)
        if i == 0:
            all_data = data
        else:
            all_data = np.concatenate((all_data, data), axis=0)
        all_label.append(label)
    dic = {}
    dic['data'] = all_data
    dic['label'] = all_label
    with open('../data/fer2013/fer2013_test.p', 'wb') as f:
        pickle.dump(dic, f)


def face_batch_align():
    data_path = '../data/fer2013/fer2013_train.p'
    bb = dlib.rectangle(0, 0, 48, 48)
    predictor = '../model/shape_predictor_68_face_landmarks.dat'
    detector = '../model/haarcascade_frontalface_default.xml'
    faceAlign = FaceAlign(predictor, detector)
    length = 48*48
    all_data = None
    all_label = None
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
        for i, d in enumerate(samples['data']):
            d = np.reshape(d, (48, 48))
            landmarks = faceAlign.predict(d, bb)
            d_a = faceAlign.align(d, landmarks)
            d_a = np.reshape(d_a, (1, length)).astype(np.uint8)
            if i == 0:
                all_data = d_a
            else:
                all_data = np.concatenate((all_data, d_a), axis=0)
            if (i+1)%20 == 0:
                print "iterations : {}".format(i)
        all_label = samples['label']
    dic = {}
    dic['data'] = all_data
    dic['label'] = all_label
    with open('../data/fer2013/fer2013_train_a.p', 'wb') as f:
        pickle.dump(dic, f)

if __name__ == "__main__":
    l = ['fer2013_train_a.p']
    dataAugmentation(l)