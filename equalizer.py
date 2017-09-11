#!/usr/bin/python3.5
import cv2
import os
from os import listdir
from os.path import isfile, join
from skimage import exposure
import numpy as np
import begin


@begin.start
def main(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    files = [f for f in listdir(source) if isfile(join(source, f)) and ".json" not in f]
    for file in files:

        img = np.asarray(cv2.imread(source + '/' + file))
        img = exposure.equalize_adapthist(img, clip_limit=0.01) * 255.0
        cv2.imwrite(destination + "/" + file, img)

