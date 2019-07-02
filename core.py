# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:57:35 2019

@author: fenezema
"""

"""
Put the libraries here
"""

import numpy as np
import cv2
import os
from random import randint
import keras
import threading
import operator
import ast
import darknet
import random
import math
import matplotlib.pyplot as plt
import time
import sys
import tensorflow as tf
import PIL.Image, PIL.ImageTk
import tkinter.font as font
import tkinter.filedialog
from tkinter import *
from ctypes import *
from multiprocessing import Process
from imutils.video import FileVideoStream
from imutils.video import FPS
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from keras.backend.tensorflow_backend import set_session