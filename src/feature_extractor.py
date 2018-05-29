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

    def crop_faces_at_same_location(self, image_path=None, image_paths=list()):
        # If one image is given
        if image_path:
            image_paths.append(image_path)

        nx = None
        ny = None
        nr = None
        for i in range(len(image_paths)):
            img = cv2.imread(image_paths[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Extract face
            x, y, w, h = self.face_cascade.detectMultiScale(gray, 1.3, 5)[0]

            # Calculate location for only the first image, for others use these values
            if i == 0:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

            # Crop the face
            face_img = img[ny:ny + nr, nx:nx + nr]

            # Yield the result or results
            yield cv2.resize(face_img, (224, 224))

    def optical_flow(self, images):
        # Crop faces
        images = [img for img in self.crop_faces_at_same_location(image_paths=images)]

        frame1 = images[0]
        previous_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        result = None
        for i in range(1, len(images)):
            frame2 = images[i]
            next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(previous_image, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            previous_image = next_img

        return result

    def spatial_features(self, image_path):
        img_array = image.img_to_array(self.crop_faces_at_same_location(image_path=image_path).__next__())
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get pre-last layer
        model_extract_features = Model(inputs=self.model.inputs, outputs=self.model.get_layer('fc2').output)

        # Extract features
        fc2_features = model_extract_features.predict(img_array)

        # Reshape the output
        fc2_features = fc2_features.reshape((4096, 1))

        return fc2_features

    @staticmethod
    def normalize_and_concat(list1, list2):
        """
        Normalize data by reducing it to 0-1 interval
        """
        list3 = []

        min1 = min(list1)
        max1 = max(list1)
        list3.extend([(i - min1) / (max1 - min1) for i in list1])

        min2 = min(list2)
        max2 = max(list2)
        list3.extend([(i - min2) / (max2 - min2) for i in list2])

        return list3

