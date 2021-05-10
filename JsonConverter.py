"""
Program to convert Json format to Yolo format (in txt)
"""

import os
from os import getcwd
from PIL import Image

# Path to dataset
dataset_path = "./dataset/"

# Path to where result should be saved
output_path = "./output/"

# Path to where old json should be saved
old_json_path = "./old_json/"

current_path = getcwd()


# Function to convert box to yolo format
def convert(width, height, box):
    d_width = 1. / width
    d_height = 1. / height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    new_width = box[1] - box[0]
    new_height = box[3] - box[2]
    x = x * d_width
    new_width = new_width * d_width
    y = y * d_height
    new_height = new_height * d_height
    return x, y, new_width, new_height


# Take list of jsons to convert
json_list = []
for file in os.listdir(dataset_path):
    if file.endswith(".json"):
        json_list.append(file)

# Converts json to yolo format
for json_name in json_list:
    yolo_name = json_name.rstrip(".json") + ".txt"
    result_path = dataset_path + json_name
    result_file = open(result_path, "r")

    yolo_path = output_path + yolo_name
    yolo_file = open(yolo_path, "a")

    lines = result_file.read().split('\n')
    for i, line in enumerate(lines):
        if "lineColor" in line:
            break
        if "label" in line:
            x1 = float(lines[i + 5].rstrip(','))
            y1 = float(lines[i + 6])
            x2 = float(lines[i + 9].rstrip(','))
            y2 = float(lines[i + 10])
            cls = line[16:17]

            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            image_path = str('%s/dataset/%s.jpg' % (current_path, os.path.splitext(json_name)[0]))

            image = Image.open(image_path)
            width = int(image.size[0])
            height = int(image.size[1])

            box = (x_min, x_max, y_min, y_max)
            bounding_box = convert(width, height, box)
            yolo_file.write(cls + " " + " ".join([str(x) for x in bounding_box]) + '\n')

    os.rename(result_path, old_json_path + json_name)
