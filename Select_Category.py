from pycocotools.coco import COCO
import requests
import os
from os.path import join
from tqdm import tqdm
import json

"""
Class to use to filter coco categories
"""


class CocoFilter:
    def __init__(self, json_path, images_path, categories='person'):
        self.json_path = json_path
        self.coco = COCO(json_path)
        self.images_path = images_path
        self.categories = categories
        self.images = self.get_images()
        self.category_ids = []

    # Function to find images with categories that were chosen
    def get_images(self):
        category_ids = self.coco.getCatIds(catNms=[self.categories])
        print("category_ids: ", category_ids)
        images_ids = self.coco.getImgIds(catIds=category_ids)
        images = self.coco.loadImgs(images_ids)
        self.category_ids = category_ids
        return images

    # Function to download coco dataset
    def download_images(self):
        os.makedirs(self.images_path, exist_ok=True)
        for im in tqdm(self.images):
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(self.images_path, im['file_name']), 'wb') as handler:
                handler.write(img_data)

    # Function that filters json file to only images with categories that were chosen
    def filter_category(self, new_json_path):
        json_oryginal = os.path.split(new_json_path)[0]
        os.makedirs(json_oryginal, exist_ok=True)
        images_ids = [i['id'] for i in self.images]
        new_images = [i for i in self.coco.dataset['images'] if i['id'] in images_ids]
        category_ids = self.category_ids
        # Filter annotations
        new_annotations = [i for i in self.coco.dataset['annotations'] if i['category_id'] in category_ids]
        # Reorganize the ids
        new_images, annotations = self.sequence_ids(new_images, new_annotations)
        # Filter categories
        new_categories = [i for i in self.coco.dataset['categories'] if i['id'] in category_ids]
        data = {
            "info": self.coco.dataset['info'],
            "licenses": self.coco.dataset['licenses'],
            "images": new_images,
            "annotations": new_annotations,
            "categories": new_categories
        }
        with open(new_json_path, 'w') as f:
            json.dump(data, f)

    # Function that changes ids of images to be in sequence
    def sequence_ids(self, images, annotations):
        images_ids = {}
        for i, image in enumerate(images):
            images_ids[images[i]['id']] = i + 1
            images[i]['id'] = i + 1
        for i, annotation in enumerate(annotations):
            annotations[i]['id'] = i + 1
            old_image_id = annotations[i]['image_id']
            annotations[i]['image_id'] = images_ids[old_image_id]
        return images, annotations


def main(dataset, year, path, categories='person'):
    json_file = join(os.path.split(path)[0], 'instances_' + dataset + year + '.json_file')
    images_path = join(path, categories + '_' + dataset)
    new_json_file = join(path, 'annotations', dataset + ".json_file")
    filter_category = CocoFilter(json_file, images_path, categ=categories)
    filter_category.download_images()
    filter_category.filter_category(new_json_file)
