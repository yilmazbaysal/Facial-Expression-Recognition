import os


def data_reader(data_path):
    label_id = 0
    for emotion in sorted(os.listdir(data_path)):
        emotion_path = os.path.join(data_path, emotion)
        label_id += 1

        for path1 in os.listdir(emotion_path):
            for path2 in os.listdir(os.path.join(emotion_path, path1)):
                images = sorted(os.listdir(os.path.join(emotion_path, path1, path2)))

                yield [os.path.join(emotion_path, path1, path2, img) for img in images], label_id - 1
