import os
from pycocotools.coco import COCO
import requests

"""
Class to keep variables about dataset
"""


class Dataset:
    def __init__(self, whole_dataset='./dataset', train_dataset='./train', train_annot='./train_annotations.json',
                 val_dataset='./val', val_annot='./val_annotations.json', *classes):
        # Path to whole dataset
        self.whole_dataset = whole_dataset

        # Path to train dataset
        self.train_dataset = train_dataset

        # Path to train annotations
        self.train_annot = train_annot

        # Path to validation dataset
        self.val_dataset = val_dataset

        # Path to validation annotations
        self.val_annot = val_annot

        # List of all categories
        self.names = []
        for i in classes:
            self.names.append(i)
        # Number of categories
        self.number_classes = len(self.names)

        # List of id of images that should be annotated by human
        self.images_annotate = []

        # List of ids of categories
        self.names_Ids = []

        # List of ids of images of categories
        self.class_images_Ids = []
        self.images = []

        # Id of last trained and annotated image
        self.last_train_id = 0
        self.last_annotate_id = 0

    # Function that uses labelme to annotate images in list
    def annotate(self):
        command = "cd " + str(self.whole_dataset)
        os.system(command)
        for i in self.images_annotate:
            command = "labelme " + str(i)
            os.system(command)

    # Function that select only images that are in categories which were chosen
    def select_class_images(self, annotations_path):
        if annotations_path:
            coco = COCO(str(annotations_path))
            for i in self.names:
                cat_Id = coco.getCatId(cat_name=[str(i)])
                self.names_Ids.append(cat_Id)
                self.class_images_Ids.append(coco.getImgId(cat_Id=cat_Id))
                self.images.append(coco.loadImgs(self.class_images_Ids))
        else:
            print("Please give annotations path")

    # Function to find images with categories that were chosen
    def get_images(self):
        class_images_Ids = self.coco.getCatIds(catNms=[self.categories])
        print("category_ids: ", class_images_Ids)
        images_ids = self.coco.getImgIds(catIds=class_images_Ids)
        images = self.coco.loadImgs(images_ids)
        self.class_images_Ids = class_images_Ids
        return images

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
