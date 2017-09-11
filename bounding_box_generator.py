#!/usr/bin/python3.5
import json
import numpy as np
from PIL import Image
import pickle
import scipy.misc
import begin
import os.path


@begin.start
def main(source, destination, bbfile, name):
    if not os.path.exists(destination):
        os.makedirs(destination)

    labels = json.load(open(bbfile))

    data = {"labels": [], "images": []}
    for label in labels:
        filename = label["filename"]
        box = label["annotations"][0]
        raw_img = Image.open(os.path.join(source, filename))

        move = [(0, 0), (box["x"] - 10, 0), (box["x"] / 2, 0), (box["x"] - 10, box["y"] - 10),
                (box["x"] / 2 - 10, box["y"] / 2 - 10), (0, box["y"] - 10), (0, box["y"] / 2 - 10),
                (box["x"] + box["width"] - raw_img.width + 10, box["y"] - 10),
                ((box["x"] + box["width"] - raw_img.width) / 2 + 10, box["y"] / 2 - 10),
                (box["x"] + box["width"] - raw_img.width + 10, 0),
                ((box["x"] + box["width"] - raw_img.width) / 2 + 10, 0),
                (box["x"] + box["width"] - raw_img.width + 10, box["y"] + box["height"] - raw_img.height + 10),
                ((box["x"] + box["width"] - raw_img.width) / 2 + 10, (box["y"] + box["height"] - raw_img.height) / 2 + 10),
                (0, box["y"] + box["height"] - raw_img.height + 10),
                (0, (box["y"] + box["height"] - raw_img.height) / 2 + 10),
                (box["x"] - 10, box["y"] + box["height"] - raw_img.height + 10),
                (box["x"] / 2 - 10, (box["y"] + box["height"] - raw_img.height) / 2 + 10)]
        for x in move:
            img = raw_img.transform(raw_img.size, Image.AFFINE, (1, 0, x[0], 0, 1, x[1]))
            new_label = (box["x"] - x[0], box["y"] - x[1], box["width"], box["height"])
            img = np.asarray(img)

            r1, r2 = img.shape[0] / 50.0, img.shape[1] / 50.0
            img = scipy.misc.imresize(img, (50, 50), interp='bilinear', mode=None)
            img = img[:, :, 1]
            new_label = (float(new_label[0]) / r2, float(new_label[1]) / r1, float(new_label[2]) / r2, float(new_label[3]) / r1)

            data["labels"].append(new_label)
            data["images"].append(img)
    pickle.dump(data, open(os.path.join(destination, name), "wb"))
