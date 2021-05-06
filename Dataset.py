import os
from pycocotools.coco import COCO
import requests


class Dataset:
    def __init__(self, whole_dataset='./dataset', train_dataset='./train', train_annot = './train_annotations.json', val_dataset='./val', val_annot = './val_annotations.json', *classes):
        self.whole_dataset = whole_dataset
        self.train_dataset = train_dataset
        self.train_annot = train_annot
        self.val_dataset = val_dataset
        self.val_annot = val_annot
        names = []
        for i in classes:
            names.append(i)
        self.number_classes = len(names)
        self.images_annotate = []
        self.names_Ids = []
        self.class_images_Ids = []
        self.images = []

    def annotate(self):
        command = "cd " + str(self.whole_dataset)
        os.system(command)
        for i in self.images_annotate:
            command = "labelme " + str(i)
            os.system(command)

    def select_class_images(self, annotations_path):
        if annotations_path:
            coco = COCO(str(annotations_path))
            for i in self.names:
                cat_Id = coco.getCatId(cat_name=[str(i)])
                self.names_Ids.append(cat_Id)
                self.class_images_Ids.append(cooc.getImgId(cat_Id=cat_Id))
                self.images.append(coco.loadImgs(imgIds))
        else:
            print("Please give annotations path")

    def filter_by_category(self, new_annot_path):
        # Filter images:
        json_parent = os.path.split(new_annot_path)[0]
        os.makedirs(json_parent, exist_ok=True)
        imgs_ids = [x['id'] for x in self.images]  # get img_ids of imgs with the category
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]
        catIds = self.class_images_Ids
        # Filter annotations
        new_annots = [x for x in self.coco.dataset['annotations'] if x['category_id'] in catIds]
        # Reorganize the ids
        new_imgs, annotations = self.modify_ids(new_imgs, new_annots)
        # Filter categories
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] in catIds]
        print("new_categories: ", new_categories)
        data = {
            "info": self.coco.dataset['info'],
            "licenses": self.coco.dataset['licenses'],
            "images": new_imgs,
            "annotations": new_annots,
            "categories": new_categories
        }
        with open(new_annot_path, 'w') as f:
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
