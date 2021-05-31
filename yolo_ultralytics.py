import os
import subprocess
import shutil
import yaml
import pathlib

from loop_utils.loop_state import *
from Dataset import Dataset


class YoloUltralytics:
    required_test_args = ["--save-txt", "--save-conf"]
    required_train_args = ["--notest", "--nosave", "--exist-ok"]

    @staticmethod
    def prepare_dataset_dir_in_correct_format(dataset: Dataset):
        pass

    @staticmethod
    def create_dataset_info_yaml(dataset: Dataset, options: dict, current_state: LoopState) -> str:
        file_contents = dict()
        file_contents.update({"train": dataset.train_dataset})
        if current_state.current_task == "detection" or current_state.current_task == "training":
            file_contents.update({"val": dataset.train_dataset})
        else:
            file_contents.update({"val": options["temp_val_dataset_dir"]})
        file_contents.update({"nc": len(dataset.names)})
        file_contents.update({"names": dataset.names})
        filename = f"{options.get('proj_dir')}/temp_dataset_info.yaml"
        os.makedirs(f"{options.get('proj_dir')}", exist_ok=True)
        with open(filename, mode="w+") as f:
            yaml.safe_dump(file_contents, f, sort_keys=False)
        return os.path.abspath(filename)

    @staticmethod
    def prepare_test_args(options: dict, dataset: Dataset, current_state: LoopState) -> list:
        args = []
        args += ["--data", YoloUltralytics.create_dataset_info_yaml(dataset, options, current_state)]
        if options.get("overwrite"):
            args += ["--exist-ok"]
        if options.get("conf_ignore_thres"):
            args += ["--conf-thres", f"{options.get('conf_ignore_thres')}"]
        args += ["--project",  f"{os.path.abspath(options.get('proj_dir'))}/{current_state.current_task}/"]
        args += ["--name", f"{current_state.current_iteration}"]
        args += ["--batch-size", f"{options.get('test_batch')}"]
        return YoloUltralytics.required_test_args + args

    @staticmethod
    def prepare_train_args(options: dict, dataset: Dataset, current_state: LoopState) -> list:
        args = []
        args += ["--data", YoloUltralytics.create_dataset_info_yaml(dataset, options, current_state)]
        args += ["--project", f"{os.path.abspath(options.get('proj_dir'))}/train/"]
        args += ["--name", f"{current_state.current_iteration}"]
        if options.get("pretrained_weights_for_first_iteration"):
            args += ["--weights", f"{os.path.abspath(options.get('pretrained_weights_for_first_iteration'))}/"]
        elif current_state.current_iteration != 0 and current_state.current_task != "init" and\
                current_state.current_task != "single_call":
            args += ["--weights", f"{os.path.abspath(options.get('proj_dir'))}"
                                  f"/train/"f"{current_state.current_iteration-1}/weights.pt"]
        args += ["--batch-size", f"{options.get('train_batch')}"]
        args += ["--epochs", f"{options.get('train_epochs')}"]
        return YoloUltralytics.required_train_args + args

    @staticmethod
    def cleanup_results(options: dict, current_state: LoopState):
        pass

    @staticmethod
    def call_test_script(options: dict, dataset: Dataset, current_state: LoopState) -> None:
        args = YoloUltralytics.prepare_test_args(options, dataset, current_state)
        owd = os.getcwd()
        os.chdir(options.get("detector_path"))
        if options.get("mute_stdout") is True:
            stdout_opt = subprocess.DEVNULL
        else:
            stdout_opt = subprocess.STDOUT
        subprocess.call(["wandb", "disabled"], stdout=stdout_opt)
        subprocess.call(["python3", "test.py", *args], stdout=stdout_opt)
        os.chdir(owd)
        for file in os.listdir(os.path.abspath(f"{options.get('proj_dir')}/{current_state.current_task}/{current_state.current_iteration}/")):
            if file != "labels":
                if pathlib.Path(os.path.abspath(f"{options.get('proj_dir')}"
                                                f"/{current_state.current_task}/{current_state.current_iteration}/{file}")).is_dir():
                    shutil.rmtree(os.path.abspath(f"{options.get('proj_dir')}"
                                                  f"/{current_state.current_task}/{current_state.current_iteration}/{file}"))
                else:
                    os.remove(os.path.abspath(f"{options.get('proj_dir')}"
                                              f"/{current_state.current_task}/{current_state.current_iteration}/{file}"))
        shutil.copytree(os.path.abspath(f"{options.get('proj_dir')}/{current_state.current_task}/"
                                    f"{current_state.current_iteration}/labels/"),
                    os.path.abspath(f"{options.get('proj_dir')}/{current_state.current_task}/{current_state.current_iteration}/"),
                        dirs_exist_ok=True)
        shutil.rmtree(os.path.abspath(f"{options.get('proj_dir')}/"
                                     f"/{current_state.current_task}/{current_state.current_iteration}/labels"))

    @staticmethod
    def call_train_script(options: dict, dataset: Dataset, current_state: LoopState) -> None:
        args = YoloUltralytics.prepare_train_args(options, dataset, current_state)
        owd = os.getcwd()
        os.chdir(options.get("detector_path"))
        if options.get("mute_stdout"):
            stdout_opt = subprocess.DEVNULL
        else:
            stdout_opt = subprocess.STDOUT
        subprocess.call(["wandb", "disabled"], stdout=stdout_opt)
        subprocess.call(["python3", "train.py", *args], stdout=stdout_opt)
        os.chdir(owd)
        shutil.copy(os.path.abspath(f"{options.get('proj_dir')}/train/"
                                    f"{current_state.current_iteration}/weights/best.pt"),
                    os.path.abspath(f"{options.get('proj_dir')}/train/{current_state.current_iteration}/weights.pt"))
        for file in os.listdir(os.path.abspath(f"{options.get('proj_dir')}/train/{current_state.current_iteration}/")):
            if file != "weights.pt":
                if pathlib.Path(os.path.abspath(f"{options.get('proj_dir')}"
                                                f"/train/{current_state.current_iteration}/{file}")).is_dir():
                    shutil.rmtree(os.path.abspath(f"{options.get('proj_dir')}"
                                                  f"/train/{current_state.current_iteration}/{file}"))
                else:
                    os.remove(os.path.abspath(f"{options.get('proj_dir')}"
                                              f"/train/{current_state.current_iteration}/{file}"))
