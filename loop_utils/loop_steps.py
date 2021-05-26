import yaml

from loop_utils.loop_state import *
from detector_handle import DetectorHandle
from Dataset import *


class LoopSteps:

    @staticmethod
    def increment_iteration_if_needed(options: dict, current_state: LoopState):
        if len(os.listdir(f"{os.path.abspath(options.get('proj_dir'))}/"
                          f"{current_state.current_task}/")) != 0 and current_state.current_iteration == 0:
            dirs = os.listdir(f"{os.path.abspath(options.get('proj_dir'))}/"f"{current_state.current_task}/")
            current_state.current_iteration = sorted(int(x) for x in dirs)[len(dirs) - 1] + 1
            return current_state.current_iteration
        else:
            return current_state.current_iteration

    @staticmethod
    def save_state(options: dict, dataset: Dataset, current_state: LoopState):
        with open(os.path.abspath(f"{options['proj_dir']}/save.yaml"), mode="w+") as f:
            save_contents = [{"last_train_id", dataset.last_train_id},
                             {"last_annotate_id", dataset.last_annotate_id}]
            yaml.dump(save_contents, f, yaml.Dumper)

    @staticmethod
    def choose_inital_training_samples(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_task = "init"
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
        current_state.current_task = "pull_new_samples"
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
        current_state.current_iteration = LoopSteps.increment_iteration_if_needed(options, current_state)
        current_state.current_task = "training"
        DetectorHandle.run_training(options, dataset, current_state)

    @staticmethod
    def validation(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_iteration = LoopSteps.increment_iteration_if_needed(options, current_state)
        current_state.current_task = "validation"
        DetectorHandle.run_detection(options, dataset, current_state)

    @staticmethod
    def detection(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_iteration = LoopSteps.increment_iteration_if_needed(options, current_state)
        current_state.current_task = "detection"
        DetectorHandle.run_detection(options, dataset, current_state)

    @staticmethod
    def select_samples(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_iteration = LoopSteps.increment_iteration_if_needed(options, current_state)
        current_state.current_task = "select_samples"
        pass

    @staticmethod
    def update_dataset(options: dict, dataset: Dataset, current_state: LoopState):
        current_state.current_iteration = LoopSteps.increment_iteration_if_needed(options, current_state)
        current_state.current_task = "update_dataset"
        pass
