from CapsNet.input_adaptor import mnist
from matplotlib import pyplot
import numpy as np

if __name__ == "__main__":
    loader = mnist.MnistLoader()

    from CapsNet.model import capsnet_test

    capsnet_test.main()
