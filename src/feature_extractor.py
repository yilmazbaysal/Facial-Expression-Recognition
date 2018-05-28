import cv2
import numpy as np
from keras import Model
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class FeatureExtractor:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            '/home/yilmaz/school/Facial-Expression-Recognition/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        self.model = VGG16(weights='imagenet', include_top=True)

    def detect_and_crop_face(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        i = 0
        result = None
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            face_img = img[ny:ny + nr, nx:nx + nr]
            result = cv2.resize(face_img, (256, 256))
            i += 1

        return cv2.resize(cv2.imread(image_path), (256, 256))

    @staticmethod
    def optical_flow(images):
        frame1 = images[0]
        previous_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        counter = 1
        result = None
        while counter < len(images):
            frame2 = images[counter]
            next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(previous_image, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            previous_image = next_img

            counter += 1

        return result

    def spatial_features(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get pre-last layer
        model_extractfeatures = Model(input=self.model.input, output=self.model.get_layer('fc2').output)

        # Extract features
        fc2_features = model_extractfeatures.predict(img_array)

        # Reshape the output
        fc2_features = fc2_features.reshape((4096, 1))

        return fc2_features
