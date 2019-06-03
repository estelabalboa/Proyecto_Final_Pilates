import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


class pose_model():

    def __init__(self):
        # Crear dataframe a partir de .csv
        df = pd.read_csv("../resources/data")


if __name__ == '__main__':
    classifier = pose_model()
