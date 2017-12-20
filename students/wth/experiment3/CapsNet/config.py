import pathlib
import os

data_folder = pathlib.Path(__file__).parent.parent.joinpath("data")
log_folder = pathlib.Path(os.getcwd()).joinpath("log")
result_folder = pathlib.Path(os.getcwd()).joinpath("result")

epoch = 50
batch_size = 128

is_training = True
