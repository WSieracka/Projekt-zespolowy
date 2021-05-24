import os
import shutil
from os import getcwd
from pycocotools.coco import COCO
import requests
from PIL import Image

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

        # List of ids of all dataset images
        self.images = []

        # Id of last trained and annotated image
        self.last_train_id = 0
        self.last_annotate_id = 0

    def classes_txt(self):
        f = open("labels.txt", "w")
        for i in self.names:
            f.write('%s\n' % i)
        f.close()

    def select_images_annotate(self, current_path, conf_threshold=0.1):
        # nr_experiment = "exp7"
        # current_path = getcwd()
        for i in self.images:
            # path = current_path+"/runs/train/"+nr_experiment+"/labels/"+str(i)+".txt"
            path = current_path + str(i) + ".txt"
            file = open(path, 'r')
            content = file.readlines()
            for line in content:
                nums = [float(j) for j in line.split(" ")]
                # 6th number is confidence
                if float(nums[5]) < conf_threshold:
                    self.images_annotate.append(i)
                    break
            file.close()

    # Function that uses labelme to annotate images in list
    def annotate(self, target_dir='/home/weronika/Pulpit/coco128/images/dataset/'):
        self.classes_txt()
        image = []
        if os.path.isdir(target_dir):
            for i in os.listdir(target_dir):
                os.remove(os.path.join(target_dir, i))
        else:
            os.system("cd ~/Pulpit/coco128/images/ ; mkdir dataset")

        for i in self.images_annotate:
            image.append(
                '/home/weronika/Pulpit/coco128/images/train2017/{images_annotate}.jpg'.format(images_annotate=i))

        for i in image:
            shutil.copy(i, target_dir)

        os.system("cd /home/weronika/Pulpit/coco128/images/ ; labelme dataset/ --labels labels.txt --autosave")

        for i in self.images_annotate:
            image.append(
                target_dir + '{images_annotate}.jpg'.format(images_annotate=i))

        for i in image:
            shutil.copy(i, self.train_dataset)

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

    def convert_json_to_txt(self, dataset_path="./dataset/", output_path="./output/", old_json_path="./old_json/"):
        current_path = getcwd()
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
