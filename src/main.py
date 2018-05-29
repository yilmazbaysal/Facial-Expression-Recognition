import cv2
import os

import numpy

from src.data_reader import data_reader
from src.feature_extractor import FeatureExtractor
from src.classifier import SingleLayerPerceptron

# To close cpp warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dimension_dict = {
    'temporal': 224 * 224 * 3,
    'spatial': 4096,
    'combined': (224 * 224 * 3) + 4096
}

#
#
#
# ################################################### TRAIN ################################################### #
print('\nExtracting features from train data\n...')

train_data = {
    'temporal_data': [],
    'temporal_labels': [],
    'spatial_data': [],
    'spatial_labels': [],
    'combined_data': [],
    'combined_labels': [],
}

label_id = None
fe = FeatureExtractor()
for images, label_id in data_reader('/home/yilmaz/school/Facial-Expression-Recognition/DATA/TRAIN'):
    # Temporal features
    optical_flow = fe.optical_flow(images).reshape(dimension_dict['temporal'])

    train_data['temporal_data'].append(optical_flow)
    train_data['temporal_labels'].append(label_id)

    # Spatial features
    for img_path in images:
        features = [f[0] for f in fe.spatial_features(img_path)]

        train_data['spatial_data'].append(features)
        train_data['spatial_labels'].append(label_id)

        # Combined features
        train_data['combined_data'].append(fe.normalize_and_concat(optical_flow, features))
        train_data['combined_labels'].append(label_id)


print('Training with temporal features\n...')
temporal_classifier = SingleLayerPerceptron(number_of_labels=label_id + 1, dimension=dimension_dict['temporal'])
temporal_classifier.train(6, numpy.array(train_data['temporal_data']), train_data['temporal_labels'])

print('Training with spatial features\n...')
spatial_classifier = SingleLayerPerceptron(number_of_labels=label_id + 1, dimension=dimension_dict['spatial'])
spatial_classifier.train(numpy.array(2, train_data['spatial_data']), train_data['spatial_labels'])

print('Training with combined (temporal and spatial) features\n...')
combined_classifier = SingleLayerPerceptron(number_of_labels=label_id + 1, dimension=dimension_dict['combined'])
combined_classifier.train(numpy.array(3, train_data['combined_data']), train_data['combined_labels'])


#
#
#
# ################################################### TEST ################################################### #
print('\nExtracting features from test data\n...')


test_data = {
    'temporal_data': [],
    'temporal_labels': [],
    'spatial_data': [],
    'spatial_labels': [],
    'combined_data': [],
    'combined_labels': [],
}

fe = FeatureExtractor()
for images, label_id in data_reader('/home/yilmaz/school/Facial-Expression-Recognition/DATA/TEST'):
    # Temporal features
    optical_flow = fe.optical_flow(images).reshape(dimension_dict['temporal'])

    test_data['temporal_data'].append(optical_flow)
    test_data['temporal_labels'].append(label_id)

    # Spatial features
    for img_path in images:
        features = [f[0] for f in fe.spatial_features(img_path)]

        test_data['spatial_data'].append(features)
        test_data['spatial_labels'].append(label_id)

        # Combined features
        test_data['combined_data'].append(fe.normalize_and_concat(optical_flow, features))
        test_data['combined_labels'].append(label_id)


print('Testing with temporal features\n...')
loss, accuracy = temporal_classifier.test(numpy.array(test_data['temporal_data']), test_data['temporal_labels'])
print('(TEMPORAL) - LOSS:', loss, '---', 'ACCURACY:', accuracy, '\n')

print('Testing with spatial features\n...')
loss, accuracy = spatial_classifier.test(numpy.array(test_data['spatial_data']), test_data['spatial_labels'])
print('(SPATIAL) - LOSS:', loss, '---', 'ACCURACY:', accuracy, '\n')

print('Testing with combined (temporal and spatial) features\n...')
loss, accuracy = combined_classifier.test(numpy.array(test_data['combined_data']), test_data['combined_labels'])
print('(COMBINED) - LOSS:', loss, '---', 'ACCURACY:', accuracy, '\n')
