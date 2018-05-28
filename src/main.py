import os

import numpy

from src.feature_extractor import FeatureExtractor
from src.recognition import Recognition

fe = FeatureExtractor()

emotions = sorted(os.listdir('/home/yilmaz/school/Facial-Expression-Recognition/DATA/TRAIN'))

r = Recognition(number_of_labels=len(emotions), dimension=256*256*3)

print(len(emotions))

data = []
labels = []

i = 0
path = '/home/yilmaz/school/Facial-Expression-Recognition/DATA/TRAIN'
for emotion in emotions:
    emotion_path = os.path.join(path, emotion)

    for path1 in os.listdir(emotion_path):
        for path2 in os.listdir(os.path.join(emotion_path, path1)):
            images = sorted(os.listdir(os.path.join(emotion_path, path1, path2)))

            images = [os.path.join(emotion_path, path1, path2, img) for img in images]

            optical_flow = fe.optical_flow([fe.detect_and_crop_face(img) for img in images])

            data.append(optical_flow.reshape(256*256*3))
            labels.append(i)

            # for img_path in images:
            #     features = fe.spatial_features(img_path)
            #
            #     data.append([x[0] for x in features])
            #     labels.append(i)
    i += 1

print('TRAIN')
r.train(numpy.array(data), labels)


####################################################


emotions = sorted(os.listdir('/home/yilmaz/school/Facial-Expression-Recognition/DATA/TEST'))

data = []
labels = []

i = 0
path = '/home/yilmaz/school/Facial-Expression-Recognition/DATA/TEST'
for emotion in emotions:
    emotion_path = os.path.join(path, emotion)

    for path1 in os.listdir(emotion_path):
        for path2 in os.listdir(os.path.join(emotion_path, path1)):
            images = sorted(os.listdir(os.path.join(emotion_path, path1, path2)))

            images = [os.path.join(emotion_path, path1, path2, img) for img in images]

            optical_flow = fe.optical_flow([fe.detect_and_crop_face(img) for img in images])

            data.append(optical_flow.reshape(256*256*3))
            labels.append(i)

            # for img_path in images:
            #     features = fe.spatial_features(img_path)
            #
            #     data.append([x[0] for x in features])
            #     labels.append(i)
    i += 1

print('TEST')

loss, accuracy = r.test(numpy.array(data), labels)

print('LOSS:', loss, '- ACCURACY:', accuracy)
#
#
# cv2.imshow('1', fe.detect_and_crop_face('/home/yilmaz/Desktop/DATA/TRAIN/anger/S011/004/S011_004_00000018.png'))
# cv2.imshow('2', fe.detect_and_crop_face('/home/yilmaz/Desktop/DATA/TRAIN/anger/S011/004/S011_004_00000019.png'))
# cv2.imshow('3', fe.detect_and_crop_face('/home/yilmaz/Desktop/DATA/TRAIN/anger/S011/004/S011_004_00000020.png'))
# cv2.imshow('4', fe.detect_and_crop_face('/home/yilmaz/Desktop/DATA/TRAIN/anger/S011/004/S011_004_00000021.png'))
#
#
# cv2.imshow('t', test)
# cv2.waitKey(0)