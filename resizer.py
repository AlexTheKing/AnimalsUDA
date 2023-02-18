import os
from PIL import Image
from tqdm import tqdm

class Resizer:
    def __init__(self, directory, resize_size=(224, 224)):
        self.directory = directory
        self.resize_size = resize_size

    def resize(self, new_directory):
        for file in tqdm(os.listdir(self.directory)):
            if not file.startswith('.'):
                Image.open(os.path.join(self.directory, file)).resize(
                    self.resize_size
                ).save(os.path.join(new_directory, file))
        return new_directory

def create_resized():
    train_folder = 'train_features'
    test_folder = 'test_features'
    new_train_folder = 'train_features_resized'
    new_test_folder = 'test_features_resized'

    try:
        os.mkdir(new_train_folder)
        resizer = Resizer(train_folder)
        resizer.resize(new_train_folder)
    except OSError as e:
        print(e)

    try:
        os.mkdir(new_test_folder)
        resizer = Resizer(test_folder)
        resizer.resize(new_test_folder)
    except OSError as e:
        print(e)

    return new_train_folder, new_test_folder