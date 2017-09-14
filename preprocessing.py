#!/usr/bin/python3.5

import os

print("removing black bands...")

os.system("./crop_images.py data/raw/broken data/raw/broken/crop")
os.system("./crop_images.py data/raw/normal data/raw/normal/crop")

print("equalizing all images...")

os.system("./equalizer.py data/raw/broken/crop data/interim/equalized/broken")
os.system("./equalizer.py data/raw/normal/crop data/interim/equalized/normal")

exit(0)

print("extracting images (crop)...")

os.system("./image_extractor.py data/interim/equalized/broken data/interim/zoom_level_1/broken bounding_box.json")
os.system("./image_extractor.py data/interim/equalized/normal data/interim/zoom_level_1/normal bounding_box.json")

os.system("./image_extractor.py data/interim/zoom_level_1/broken data/interim/zoom_level_2/broken bounding_box.json")
os.system("./image_extractor.py data/interim/zoom_level_1/normal data/interim/zoom_level_2/normal bounding_box.json")


print("generating bounding box data set...")

os.system("./bounding_box_generator.py data/interim/equalized/broken " +
          "data/interim/bounding_box data/interim/equalized/broken/bounding_box.json broken_bg")
os.system("./bounding_box_generator.py data/interim/equalized/normal " +
          "data/interim/bounding_box data/interim/equalized/normal/bounding_box.json normal_bg")

os.system("./bounding_box_generator.py data/interim/zoom_level_1/broken " +
          "data/interim/bounding_box data/interim/zoom_level_1/broken/bounding_box.json broken_bg")
os.system("./bounding_box_generator.py data/interim/zoom_level_1/normal " +
          "data/interim/bounding_box data/interim/zoom_level_1/normal/bounding_box.json normal_bg")
