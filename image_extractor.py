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
    labels = json.load(open(join(source, bbfile)))
    for label in labels:
        filename = label["filename"]
        box = label["annotations"][0]
        raw_img = Image.open(join(source, filename))
        crop_img = raw_img.crop((int(box["x"]), int(box["y"]),
                                 int(box["width"]) + int(box["x"]), int(box["height"]) + int(box["y"])))
        crop_img.save(join(destination, filename))

