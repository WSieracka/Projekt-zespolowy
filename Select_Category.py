from pycocotools.coco import COCO
import requests
import os
from os.path import join
from tqdm import tqdm
import json


class coco_category_filter:
    def __init__(self, json_path, imgs_dir, categ='person'):
        self.coco = COCO(json_path)
        self.json_path = json_path
        self.imgs_dir = imgs_dir
        self.categ = categ
        self.images = self.get_imgs()

    def get_imgs(self):
        catIds = self.coco.getCatIds(catNms=[self.categ])
        print("catIds: ", catIds)
        imgIds = self.coco.getImgIds(catIds=catIds)
        images = self.coco.loadImgs(imgIds)
        self.catIds = catIds
        return images

    def save_imgs(self):
        os.makedirs(self.imgs_dir, exist_ok=True)
        for im in tqdm(self.images):
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(self.imgs_dir, im['file_name']), 'wb') as handler:
                handler.write(img_data)

    def filter_by_category(self, new_json_path):
        json_parent = os.path.split(new_json_path)[0]
        os.makedirs(json_parent, exist_ok=True)
        imgs_ids = [x['id'] for x in self.images]  # get img_ids of imgs with the category
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]
        catIds = self.catIds
        # Filter annotations
        new_annots = [x for x in self.coco.dataset['annotations'] if x['category_id'] in catIds]
        # Reorganize the ids
        new_imgs, annotations = self.modify_ids(new_imgs, new_annots)
        # Filter categories
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] in catIds]
        data = {
            "info": self.coco.dataset['info'],
            "licenses": self.coco.dataset['licenses'],
            "images": new_imgs,
            "annotations": new_annots,
            "categories": new_categories
        }
        with open(new_json_path, 'w') as f:
            json.dump(data, f)

    def modify_ids(self, images, annotations):
        old_new_imgs_ids = {}
        for n, im in enumerate(images):
            old_new_imgs_ids[images[n]['id']] = n + 1
            images[n]['id'] = n + 1
        for n, ann in enumerate(annotations):
            annotations[n]['id'] = n + 1
            old_image_id = annotations[n]['image_id']
            annotations[n]['image_id'] = old_new_imgs_ids[old_image_id]
        return images, annotations


def main(subset, year, root_dir, category='person'):
    json_file = join(os.path.split(root_dir)[0], 'instances_' + subset + year + '.json')
    imgs_dir = join(root_dir, category + '_' + subset)
    new_json_file = join(root_dir, 'annotations', subset + ".json")
    coco_filter = coco_category_filter(json_file, imgs_dir, categ=category)
    coco_filter.save_imgs()
    coco_filter.filter_json_by_category(new_json_file)

    # Example usage:
    # subset, year = 'train', '2017'  # val - train
    # root_dir = '/home/ubuntu/coco_person_ds/coco_dataset'
    # main(subset, year, root_dir, category='person')
