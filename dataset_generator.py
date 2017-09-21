#!/usr/bin/python3.5

import scipy.ndimage
import scipy.misc
import pickle

import numpy as np
import lib
import os.path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels_ = ["broken", "normal"]

rsource = "data/interim/generated/faces/"
filename = ["train", "test"]

directories = ["data/interim/generated/eq_faces", "data/interim/generated/faces",
               "data/interim/generated/r_faces", "data/interim/generated/req_faces"]
images = []
labels = []

data = dict({k: {} for k in labels_})

for directory in directories:

    for label in labels_:

        source = os.path.join(directory, label)
        files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f)) and ".json" not in f]

        for file in files:
            if file not in data[label]:
                data[label][file] = {"images": [], "labels": []}
            image = scipy.ndimage.imread(os.path.join(directory, label, file), flatten=True)
            image = scipy.misc.imresize(image, (100, 100), interp='bilinear', mode=None)
            data[label][file]['images'].append(image)
            data[label][file]['labels'].append((0, 1) if label == "broken" else (1, 0))

if not os.path.exists("data/interim/classifier/broken"):
    for x in labels_:
        os.makedirs(os.path.join("data/interim/classifier", x))

for label in labels_:
    for k, x in data[label].items():
        pickle.dump(x, open(os.path.join("data/interim/classifier", label, os.path.splitext(k)[0]), "wb"))

exit(0)


images = np.array(images)
labels = np.array(labels)

images, labels = lib.data.shuffle(images, labels)


pickle.dump({"images": images[int(0.1 * len(images)):],
             "labels": labels[int(0.1 * len(labels)):]}, open("data/interim/classifier/train", "wb"))

pickle.dump({"images": images[:int(0.1 * len(images))],
             "labels": labels[:int(0.1 * len(labels))]}, open("data/interim/classifier/test", "wb"))


