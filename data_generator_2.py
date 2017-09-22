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
    all_files = list(glob.iglob('data/interim/classifier/broken/*', recursive=True))
    all_files2 = list(glob.iglob('data/interim/classifier/normal/*', recursive=True))

    random.shuffle(all_files)
    test = all_files[:int(0.1 * len(all_files))] + all_files2[:int(0.1 * len(all_files))]
    train = all_files[int(0.1 * len(all_files)):] + all_files2[int(0.1 * len(all_files)):]
    make_dataset(test, "data/interim/classifier/test")
    make_dataset(train, "data/interim/classifier/train")

