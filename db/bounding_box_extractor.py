#!/usr/bin/python3.5

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import begin
import psycopg2
import psycopg2.extras
from os.path import join
import PIL.Image
import pickle
import os
import numpy as np
import scipy.misc


def translate_context(box, context):
    t = {k: box[k] - context[k] for k in box.keys()}
    move = [(0, 0), (t["x"], 0), (t["x"] / 2, 0), (t["x"], t["y"]),
            (t["x"] / 2, t["y"] / 2), (0, t["y"]), (0, t["y"] / 2), (t["x"] + t["w"], t["y"]),
            ((t["x"] + t["w"]) / 2, t["y"] / 2), (t["x"] + t["w"], 0), ((t["x"] + t["w"]) / 2, 0),
            (t["x"] + t["w"], t["y"] + t["h"]), ((t["x"] + t["w"]) / 2, (t["y"] + t["h"]) / 2),
            (0, t["y"] + t["h"]), (0, (t["y"] + t["h"]) / 2), (t["x"], t["y"] + t["h"]),
            (t["x"] / 2, (t["y"] + t["h"]) / 2)]
    return move


@begin.start
def main(debug=None):
    if not os.path.exists("../data/interim/bounding_box"):
        os.makedirs("../data/interim/bounding_box")
        os.makedirs("../data/interim/bounding_box/broken")
        os.makedirs("../data/interim/bounding_box/normal")

    query = "SELECT images.filename as filename, images.id as id, images.details->>'ACL' as folder, " \
            "F.settings as face " + \
            "FROM boundingbox as F " + \
            "INNER JOIN images ON images.id = F.fk_image_id"

    conn = psycopg2.connect("dbname='watson' user='sebastien' host='localhost' password='password'")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(query)
    boxes = cur.fetchall()

    if debug:
        plt.ion()
        fig, ax = plt.subplots(1)

    for box in boxes:
        data = {"labels": [], "images": []}
        img = PIL.Image.open("../data/interim/equalized/" + join(box["folder"], box["filename"]))
        img2 = PIL.Image.open("../data/raw/" + join(box["folder"], "crop", box["filename"]))
        imgs = [img, img2]
        face = {k[0]: box["face"][k] for k in ["x", "y", "height", "width"]}
        face = {**{k: face[k] * 0.7 for k in ["x", "y"]}, **{k: face[k] * 1.4 for k in ["h", "w"]}}
        zooms = [face]
        for img in imgs:
            context = {"x": 0, "y": 0, "w": img.size[0], "h": img.size[1]}
            for x in range(len(zooms)):
                cnt = translate_context(zooms[x], context)
                for t in cnt:
                    tmp_img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, t[0], 0, 1, t[1]))
                    tmp_img = tmp_img.crop((int(context["x"]), int(context["y"]),
                                            int(context["x"] + context["w"]),
                                            int(context["y"] + context["h"])))
                    tmp_img = np.asarray(tmp_img)
                    label = {"x": zooms[x]["x"] - context["x"] - t[0],
                             "y": zooms[x]["y"] - context["y"] - t[1],
                             "w": zooms[x]["w"], "h": zooms[x]["h"]}

                    r1, r2 = tmp_img.shape[0] / 50.0, tmp_img.shape[1] / 50.0
                    tmp_img = scipy.misc.imresize(tmp_img, (50, 50), interp='bilinear', mode=None)[:, :, 1]
                    label = {k: label[k] / (r1 if k in ["x", "y"] else r2) for k in label.keys()}
                    if debug:
                        ax.clear()
                        ax.imshow(tmp_img)
                        rect = patches.Rectangle((label["x"], label["y"]), label["w"], label["h"], linewidth=1,
                                                 edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.draw()
                        plt.pause(0.1)
                    if not debug:
                        data["images"].append(tmp_img)
                        data["labels"].append((label["x"], label["y"], label["w"], label["h"]))
                context = zooms[x]
        if not debug:
            pickle.dump(data, open(join("../data/interim/bounding_box", join(box["folder"], str(box["id"]))), "wb"))

