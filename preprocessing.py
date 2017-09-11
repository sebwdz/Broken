#!/usr/bin/python3.5

import os

print("removing black bands...")

os.system("./crop_images.py data/raw/package_1/broken data/raw/package_1/broken/crop")
os.system("./crop_images.py data/raw/package_1/normal data/raw/package_1/normal/crop")
os.system("./crop_images.py data/raw/package_2/broken data/raw/package_2/broken/crop")
os.system("./crop_images.py data/raw/package_2/normal data/raw/package_2/normal/crop")

print("equalizing all images...")

os.system("./equalizer.py data/raw/package_1/broken/crop data/interim/package_1/equalized/broken")
os.system("./equalizer.py data/raw/package_1/normal/crop data/interim/package_1/equalized/normal")
os.system("./equalizer.py data/raw/package_2/broken/crop data/interim/package_2/equalized/broken")
os.system("./equalizer.py data/raw/package_2/normal/crop data/interim/package_2/equalized/normal")

print("extracting images (crop)...")

os.system("./image_extractor.py data/interim/package_1/equalized/broken " +
          "data/interim/package_1/zoom_level_1/broken bounding_box.json")
os.system("./image_extractor.py data/interim/package_1/equalized/normal " +
          "data/interim/package_1/zoom_level_1/normal bounding_box.json")
os.system("./image_extractor.py data/interim/package_2/equalized/broken " +
          "data/interim/package_2/zoom_level_1/broken bounding_box.json")
os.system("./image_extractor.py data/interim/package_2/equalized/normal " +
          "data/interim/package_2/zoom_level_1/normal bounding_box.json")

os.system("./image_extractor.py data/interim/package_1/zoom_level_1/broken " +
          "data/interim/package_1/zoom_level_2/broken bounding_box.json")
os.system("./image_extractor.py data/interim/package_1/zoom_level_1/normal " +
          "data/interim/package_1/zoom_level_2/normal bounding_box.json")
os.system("./image_extractor.py data/interim/package_2/zoom_level_1/broken " +
          "data/interim/package_1/zoom_level_2/broken bounding_box.json")
os.system("./image_extractor.py data/interim/package_2/zoom_level_1/normal " +
          "data/interim/package_1/zoom_level_2/normal bounding_box.json")

print("generating bounding box data set...")

os.system("./bounding_box_generator.py data/interim/package_1/equalized/broken " +
          "data/interim/package_1/bounding_box data/interim/package_1/equalized/broken/bounding_box.json broken_bg")
os.system("./bounding_box_generator.py data/interim/package_1/equalized/normal " +
          "data/interim/package_1/bounding_box data/interim/package_1/equalized/normal/bounding_box.json normal_bg")
os.system("./bounding_box_generator.py data/interim/package_2/equalized/broken " +
          "data/interim/package_2/bounding_box data/interim/package_2/equalized/broken/bounding_box.json broken_lt")
os.system("./bounding_box_generator.py data/interim/package_2/equalized/normal " +
          "data/interim/package_2/bounding_box data/interim/package_2/equalized/normal/bounding_box.json normal_lt")

os.system("./bounding_box_generator.py data/interim/package_1/zoom_level_1/broken " +
          "data/interim/package_1/bounding_box data/interim/package_1/zoom_level_1/broken/bounding_box.json broken_bg")
os.system("./bounding_box_generator.py data/interim/package_1/zoom_level_1/normal " +
          "data/interim/package_1/bounding_box data/interim/package_1/zoom_level_1/normal/bounding_box.json normal_bg")
os.system("./bounding_box_generator.py data/interim/package_2/zoom_level_1/broken " +
          "data/interim/package_1/bounding_box data/interim/package_2/zoom_level_1/broken/bounding_box.json broken_lt")
os.system("./bounding_box_generator.py data/interim/package_2/zoom_level_1/normal " +
          "data/interim/package_1/bounding_box data/interim/package_2/zoom_level_1/normal/bounding_box.json normal_lt")