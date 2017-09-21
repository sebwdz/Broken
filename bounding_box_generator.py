#!/usr/bin/python3.5
import pickle
import begin
import glob
import random


def make_dataset(files, destination):
    data = {"images": [], "labels": []}
    for file in files:
        tmp = pickle.load(open(file, "rb"))
        data["images"] += tmp["images"]
        data["labels"] += tmp["labels"]
    pickle.dump(data, open(destination, "wb"))


@begin.start
def main():
    all_files = list(glob.iglob('data/interim/bounding_box/*/*', recursive=True))
    random.shuffle(all_files)
    test = all_files[:int(0.05 * len(all_files))]
    train = all_files[int(0.05 * len(all_files)):]
    print(len(test))
    print(len(train))
    make_dataset(test, "data/interim/bounding_box/test")
    make_dataset(train, "data/interim/bounding_box/train")

