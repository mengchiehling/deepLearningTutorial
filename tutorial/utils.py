import random
from glob import glob
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from imgaug.augmenters import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder

from tutorial.io.get_path_definition import get_project_dir


class DataGenerator(Sequence):

    def __init__(self, dir: str, sample_size: Optional[int] = None, batch_size: int = 32, shuffle: bool = True,
                 image_augmentation: Optional[Sequential] = None):

        """
        Args:
            dir: directory in which images are stored
            sample_size: Optional; number of images will be sampled in each of sub_directory,
            if not provided all images in the dir are taken into account.
            batch_size: number of images in each of batch
            shuffle: if shuffle the order of the data
        """

        self.dir = dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_size = sample_size
        self.image_augmentation = image_augmentation  # new line

        self.on_epoch_end()

        self.max = self.__len__()
        self.n = 0

    def __transform_to_dataframe(self) -> pd.DataFrame:

        """
        transform the data into a pandas dataframe to track the image files and the corresponding labels
        """

        dirs = glob(f"{get_project_dir()}/data/{self.dir}/*")

        data = []

        for dir in dirs:

            files = glob(f"{dir}/*.png")
            if self.sample_size:  # modification
                if not self.image_augmentation:
                    sampled_files = random.sample(files, min(self.sample_size, len(files)))  # no repetition
                else:
                    sampled_files = random.choices(files, k=self.sample_size)  # repetition can take place
            else:
                sampled_files = files

            label = int(dir.split("/")[-1])

            for f in sampled_files:
                data.append([f, label])

        df = pd.DataFrame(data=data, columns=['filepath', 'label'], dtype=object)

        return df

    def on_epoch_end(self):

        self.df = self.__transform_to_dataframe()
        self.indices = self.df.index.tolist()

        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        #  Denotes the number of batches per epoch
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        # Generate data
        X, y = self.get_data(batch)

        return X, y

    def get_data(self, batch: List) -> Tuple[np.ndarray, np.ndarray]:

        df_batch = self.df.loc[batch]

        image_dataset = []
        labels = []

        for _, row in df_batch.iterrows():
            f = row['filepath']
            if not self.image_augmentation:
                image_dataset.append(np.array(Image.open(f)) / 255.0)
            else:
                image_dataset.append(self.image_augmentation.augment_image(np.array(Image.open(f))) / 255.0)
            labels.append(row['label'])

        return np.array(image_dataset), np.array(labels)

    def __next__(self):

        """
        generate data of size batch_size
        """

        if self.n >= self.max:
            self.n = 0

        result = self.__getitem__(self.n)
        self.n += 1
        return result


class DataGeneratorClassification(DataGenerator):

    def __init__(self, dir: str, sample_size: Optional[int] = None, batch_size: int = 32, shuffle: bool = True,
                 image_augmentation: Optional[Sequential] = None):

        super().__init__(dir=dir, sample_size=sample_size, batch_size=batch_size, shuffle=shuffle,
                         image_augmentation=image_augmentation)

        self.encoder = OneHotEncoder()

        self.fit_encoder()

    def fit_encoder(self):

        dirs = glob(f"{get_project_dir()}/data/{self.dir}/*")
        labels = [int(dir.split("/")[-1]) for dir in dirs]

        self.encoder.fit(np.array(labels).reshape(-1, 1))

    def get_data(self, batch: List) -> Tuple[np.ndarray, np.ndarray]:

        # Overwrite

        df_batch = self.df.loc[batch]

        image_dataset = []
        labels = []

        for _, row in df_batch.iterrows():
            f = row['filepath']
            if not self.image_augmentation:
                image_dataset.append(np.array(Image.open(f)) / 255.0)
            else:
                image_dataset.append(self.image_augmentation.augment_image(np.array(Image.open(f))) / 255.0)
            labels.append(row['label'])

        return np.array(image_dataset), self.encoder.transform(np.array(labels).reshape(-1, 1))


