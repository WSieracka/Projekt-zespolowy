import os
import shutil

images_annotate = {"000000000009", "000000000025", "000000000030"}
image = []
target_dir = '/home/weronika/Pulpit/coco128/images/dataset/'
if os.path.isdir(target_dir):
    for i in os.listdir(target_dir):
        os.remove(os.path.join(target_dir, i))
else:
    os.system("cd ~/Pulpit/coco128/images/ ; mkdir dataset")

for i in images_annotate:
    image.append('/home/weronika/Pulpit/coco128/images/train2017/{images_annotate}.jpg'.format(images_annotate=i))

for i in image:
    shutil.copy(i, target_dir)

os.system("cd /home/weronika/Pulpit/coco128/images/ ; labelme dataset/ --labels labels.txt --autosave")
