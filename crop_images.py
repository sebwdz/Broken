#!/usr/bin/python3.5
import cv2
import os
from os import listdir
from os.path import isfile, join
import begin


@begin.start
def main(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    files = [f for f in listdir(source) if isfile(join(source, f))]
    for file in files:
        img = cv2.imread(source + '/' + file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        _, contours, hierarchy, = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        big = (0, 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > big[0]:
                big = (w * h, (x, y, w, h))

        im = img[big[1][1]:big[1][1] + big[1][3], big[1][0]:big[1][0] + big[1][2]]

        cv2.imwrite(destination + '/' + file, im)

