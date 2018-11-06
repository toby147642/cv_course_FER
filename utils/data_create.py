import csv
import numpy as np
import cv2
import time
import os
if __name__ == "__main__":
    data = []
    train_path = './data/fer2013/train'
    test_path = './data/fer2013/test'
    with open('./data/fer2013.csv', 'r') as f:
        reader = csv.reader(f)
        for t in reader:
            data.append(t)
    for i, line in enumerate(data):
        if i == 0:
            continue
        print 'number : {}'.format(i)
        raw_data = [eval(line[1].split()[j]) for j in range(2304)]
        raw_data = np.array(raw_data, dtype=np.uint8)
        raw_data = np.reshape(raw_data, (48, 48))
        raw_label = line[0]
        im_name = 'fer_{}_{}.bmp'.format(time.time(), raw_label)
        if line[2] == 'Training':
            cv2.imwrite(os.path.join(train_path, im_name), raw_data)
        else:
            cv2.imwrite(os.path.join(test_path, im_name), raw_data)
        print 'save as {}'.format(im_name)