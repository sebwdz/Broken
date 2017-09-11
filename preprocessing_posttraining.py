#!/usr/bin/python3.5

import os


print("generating bounding box zoom_level_1 -> zoom_level_2...")

os.system("./bounding_box_extractor.py data/interim/package_1/equalized/normal "
          "data/interim/package_1/generated/zoom_level_1/normal")
os.system("./bounding_box_extractor.py data/interim/package_1/equalized/broken "
          "data/interim/package_1/generated/zoom_level_1/broken")
os.system("./bounding_box_extractor.py data/interim/package_2/equalized/normal "
          "data/interim/package_2/generated/zoom_level_1/normal")
os.system("./bounding_box_extractor.py data/interim/package_2/equalized/broken "
          "data/interim/package_2/generated/zoom_level_1/broken")

os.system("./bounding_box_generator.py data/interim/package_1/generated/zoom_level_1/broken " +
          "data/interim/package_1/bounding_box data/interim/package_1/zoom_level_1/broken/bounding_box.json "
          "broken_lt_g")
os.system("./bounding_box_generator.py data/interim/package_1/zoom_level_1/normal " +
          "data/interim/package_1/bounding_box data/interim/package_1/zoom_level_1/normal/bounding_box.json "
          "normal_lt_g")

os.system("./bounding_box_generator.py data/interim/package_2/generated/zoom_level_1/broken " +
          "data/interim/package_2/bounding_box data/interim/package_2/zoom_level_1/broken/bounding_box.json "
          "broken_lt_g")
os.system("./bounding_box_generator.py data/interim/package_2/zoom_level_1/normal " +
          "data/interim/package_2/bounding_box data/interim/package_2/zoom_level_1/normal/bounding_box.json "
          "normal_lt_g")

os.system("./bounding_box_extractor.py data/interim/package_1/equalized/normal "
          "data/interim/package_1/generated/zoom_level_1/normal")
os.system("./bounding_box_extractor.py data/interim/package_1/equalized/broken "
          "data/interim/package_1/generated/zoom_level_1/broken")
os.system("./bounding_box_extractor.py data/interim/package_2/equalized/normal "
          "data/interim/package_2/generated/zoom_level_1/normal")
os.system("./bounding_box_extractor.py data/interim/package_2/equalized/broken "
          "data/interim/package_2/generated/zoom_level_1/broken")

os.system("./bounding_box_extractor.py data/interim/package_1/generated/zoom_level_1/normal "
          "data/interim/package_1/generated/zoom_level_2g/normal")
os.system("./bounding_box_extractor.py data/interim/package_1/generated/zoom_level_1/broken "
          "data/interim/package_1/generated/zoom_level_2g/broken")
os.system("./bounding_box_extractor.py data/interim/package_2/generated/zoom_level_1/normal "
          "data/interim/package_2/generated/zoom_level_2g/normal")
os.system("./bounding_box_extractor.py data/interim/package_2/generated/zoom_level_1/broken "
          "data/interim/package_2/generated/zoom_level_2g/broken")
