import os

import yaml
import subprocess

from loop_utils.loop_state import *
from detector_handle import DetectorHandle
from Dataset import *


class LoopSteps:
    @staticmethod
    def finish_iteration(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_iteration = current_state.current_iteration + 1
        LoopSteps.save_state(options, dataset, current_state)

    @staticmethod
    def save_state(options: dict, dataset: Dataset, current_state: LoopState):
        os.makedirs(f"{options.get('proj_dir')}", exist_ok=True)
        with open(os.path.abspath(f"{options['proj_dir']}/save.yaml"), mode="w+") as f:
            save_contents = {"last_train_id": dataset.last_train_id,
                             "last_annotate_id": dataset.last_annotate_id,
                             "last_task": current_state.current_task,
                             "iteration": current_state.current_iteration}
            yaml.safe_dump(save_contents, f)

    @staticmethod
    def read_save_if_possible(options: dict, dataset: Dataset, current_state: LoopState):
        if os.path.exists(os.path.abspath(f"{options['proj_dir']}/save.yaml")):
            with open(os.path.abspath(f"{options['proj_dir']}/save.yaml"), mode="r") as f:
                state_dict = yaml.load(f, yaml.SafeLoader)
                loaded_state = LoopState(state_dict["last_task"], state_dict["iteration"])
                loaded_dataset = Dataset(dataset.whole_dataset, dataset.train_dataset, dataset.train_annot,
                                         dataset.val_dataset, dataset.val_annot, *dataset.names)
                loaded_dataset.last_train_id = state_dict["last_train_id"]
                loaded_dataset.last_annotate_id = state_dict["last_annotate_id"]
                return loaded_dataset, loaded_state

    @staticmethod
    def choose_initial_training_samples(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "training"
        if not os.path.exists(os.path.abspath(dataset.train_dataset)):
            os.makedirs(os.path.abspath(dataset.train_dataset))
        if not os.path.exists(os.path.abspath(dataset.train_annot)):
            os.makedirs(os.path.abspath(dataset.train_annot))
        all_images = os.listdir(dataset.whole_dataset)
        ids = [id.strip(".txt") for id in all_images]
        num_images = int(options["training_size_incrementation"]*len(ids))
        dataset.last_train_id = num_images - 1
        for file in os.listdir(os.path.abspath(dataset.whole_dataset))[0:num_images]:
            shutil.copy(os.path.abspath(f"{dataset.whole_dataset}/{file}"), os.path.abspath(dataset.train_annot))
        whole_dataset_images = options["whole_dataset_images_dir"]
        for file in os.listdir(os.path.abspath(whole_dataset_images))[0:num_images]:
            shutil.copy(os.path.abspath(f"{whole_dataset_images}/{file}"), os.path.abspath(dataset.train_dataset))

    @staticmethod
    def choose_new_training_samples(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "training"
        if not os.path.exists(os.path.abspath(dataset.train_dataset)):
            os.makedirs(os.path.abspath(dataset.train_dataset))
        if not os.path.exists(os.path.abspath(dataset.train_annot)):
            os.makedirs(os.path.abspath(dataset.train_annot))
        all_images = os.listdir(dataset.whole_dataset)
        ids = [id.strip(".txt") for id in all_images]
        num_images = int(options["training_size_incrementation"]*len(ids))
        dataset.last_train_id = num_images - 1
        for file in os.listdir(os.path.abspath(dataset.whole_dataset))[dataset.last_train_id:num_images]:
            shutil.copy(os.path.abspath(f"{dataset.whole_dataset}/{file}"), os.path.abspath(dataset.train_annot))
        whole_dataset_images = options["whole_dataset_images_dir"]
        for file in os.listdir(os.path.abspath(whole_dataset_images))[dataset.last_train_id:num_images]:
            shutil.copy(os.path.abspath(f"{whole_dataset_images}/{file}"), os.path.abspath(dataset.train_dataset))

    @staticmethod
    def training(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "training"
        DetectorHandle.run_training(options, dataset, current_state)

    @staticmethod
    def validation(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "validation"

        options['temp_val_dataset_dir'] = os.path.abspath(options['temp_val_dataset_dir'])
        options['temp_val_dataset_dir'] = os.path.abspath(options['temp_val_dataset_dir'])

        os.makedirs(options['temp_val_dataset_dir'], exist_ok=True)
        os.makedirs(options['temp_val_labels'], exist_ok=True)

        val_names = os.listdir(os.path.abspath(dataset.val_dataset))

        if options["validation_samples_amount"] > len(val_names):
            val_amount = len(val_names)
        else:
            val_amount = options["validation_samples_amount"]

        val_names = val_names[0:val_amount - 1]

        for file in val_names:
            shutil.copy(os.path.abspath(f"{dataset.val_dataset}/{file}"), os.path.abspath(options['temp_val_dataset_dir']+ f"/{file}"))

        val_names = os.listdir(os.path.abspath(dataset.val_annot))[0:val_amount - 1]

        for file in val_names:
            shutil.copy(os.path.abspath(f"{dataset.val_annot}/{file}"), os.path.abspath(options['temp_val_labels'] + f"/{file}"))
        DetectorHandle.run_detection(options, dataset, current_state)

    @staticmethod
    def detection(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "detection"
        DetectorHandle.run_detection(options, dataset, current_state)

    @staticmethod
    def select_samples(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "select_samples"
        dataset.images_annotate = []
        names = os.listdir(f"{options['proj_dir']}/detection/{current_state.current_iteration}")
        for name in names:
            name = name.strip(".txt")
            dataset.images.append(name)
        dataset.select_images_annotate(os.path.abspath(f"{options['proj_dir']}/detection/{current_state.current_iteration}/"),
                                       conf_threshold=options["conf_sample_thres"])
        labelme_temp_dir = os.path.abspath(f"{options['proj_dir']}/labelme_temp")
        if os.path.exists(labelme_temp_dir):
            shutil.rmtree(labelme_temp_dir)
        os.makedirs(labelme_temp_dir)
        owd = os.getcwd()
        os.chdir(options["proj_dir"])
        dataset.classes_txt()
        os.chdir(owd)
        for image in dataset.images_annotate:
            shutil.copy(os.path.abspath(f"{options['train_dataset_dir']}/{image}.jpg"), os.path.abspath(labelme_temp_dir))
            subprocess.call(["labelme", os.path.abspath(labelme_temp_dir), "--autosave", "--labels", f"{os.path.abspath(options['proj_dir'])}/labels.txt"])
