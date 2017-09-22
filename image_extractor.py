#!/usr/bin/python3.5
import os
from os.path import join
import begin
import json
import PIL.Image as Image


@begin.start
def main(source, destination, bbfile):
    if not os.path.exists(destination):
        os.makedirs(destination)
    labels = json.load(open(bbfile))
    for label in labels:
        filename = label["filename"]
        box = label["annotations"][0]
        if box['class'] != 'Face':
            box = label["annotations"][1]
        raw_img = Image.open(join(source, filename))
        crop_img = raw_img.crop((int(box["x"] * 0.7), int(box["y"] * 0.7),
                                 int(box["width"] * 1.4) + int(box["x"] * 0.7),
                                 int(box["height"] * 1.4) + int(box["y"] * 0.7)))
        crop_img.save(join(destination, filename))

