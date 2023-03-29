""" init """
import sys
import os
import pathlib

TRAIN_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)  # "../../"
sys.path.insert(0, TRAIN_PATH)
