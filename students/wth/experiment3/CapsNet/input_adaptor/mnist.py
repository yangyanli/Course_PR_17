from . import InputAdaptor
from ..config import data_folder
from PIL import Image
import numpy as np


class MnistLoader(InputAdaptor):
    def __init__(self,
                 image_filename=data_folder.joinpath("MNIST").joinpath("train-images.idx3-ubyte"),
                 label_filename=data_folder.joinpath("MNIST").joinpath("train-labels.idx1-ubyte"),
                 in_memory=False,
                 shuffle=False):

        self.in_memory = in_memory
        if shuffle:
            self.in_memory = True
        self.shuffle = shuffle

        self._image_width = 28
        self._image_height = 28
        self._label_length = 1

        self.read_pointer = 0

        self.image_file = open(image_filename, "rb")
        self.label_file = open(label_filename, "rb")

        # check magic
        magic = int.from_bytes(self.label_file.read(4), byteorder="big")
        if magic != 2049:
            raise IOError("Label file magic mismatch.")

        magic = int.from_bytes(self.image_file.read(4), byteorder="big")
        if magic != 2051:
            raise IOError("Image file magic mismatch.")

        # check data set size
        self._size = int.from_bytes(self.label_file.read(4), byteorder="big")
        size2 = int.from_bytes(self.image_file.read(4), byteorder="big")

        if self._size != size2:
            raise IOError("Image and label size mismatch.")

        img_w = int.from_bytes(self.image_file.read(4), byteorder="big")
        img_h = int.from_bytes(self.image_file.read(4), byteorder="big")
        if img_w != 28 or img_h != 28:
            raise IOError("Image size is not 28*28. maybe the data set is not mnist?")

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_height(self):
        return self._image_height

    @property
    def label_length(self):
        return self._label_length

    @property
    def size(self):
        return self._size

    def next(self):
        label = int.from_bytes(self.label_file.read(1), byteorder="big")
        if not label:
            # end of file
            return None, None
        image_raw = self.image_file.read(784)

        image = np.reshape(np.frombuffer(image_raw, dtype=np.uint8), [28, 28])
        self.read_pointer += 1
        return label, image

    def batch(self, n=50):
        labels_raw = self.label_file.read(n)
        images_raw = self.image_file.read(n*784)
        if len(labels_raw) < n:
            print("[Warning][MnistLoader]: Remaining data can't fill up a batch.")
            n = labels_raw
        if n == 0:
            return [], []

        labels = [int(labels_raw[i]) for i in range(n)]
        images = [np.reshape(np.frombuffer(images_raw[i*784:(i+1)*784], dtype=np.uint8), [28, 28, 1]) for i in range(n)]

        self.read_pointer += n
        return labels, images

    def __del__(self):
        if self.image_file and not self.image_file.closed:
            self.image_file.close()
        if self.label_file and not self.label_file.closed:
            self.label_file.close()
