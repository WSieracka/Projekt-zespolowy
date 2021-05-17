import os
import subprocess


from loop import LoopState
from Dataset import Dataset


class YoloUltralytics:
    required_test_args = "--task test --save-txt --save-conf"
    required_train_args = "--notest --nosave --exist-ok"

    @staticmethod
    def create_dataset_info_yaml(dataset: Dataset):
        pass

    @staticmethod
    def prepare_test_args(options: dict, current_state: LoopState) -> str:
        proj_dir = f"--project  {os.path.abspath(options.get('proj_dir'))}/{current_state.current_task}/"
        name = f"--name {current_state.current_iteration}"
        test_batch_size = f"--batch-size {options.get('test_batch')}"
        dataset_info = f"--data {options.get('dataset_info_yaml')}"
        args = [proj_dir, name, test_batch_size, dataset_info]
        return " ".join(args).join(YoloUltralytics.required_test_args)

    @staticmethod
    def prepare_train_args(options: dict, current_state: LoopState) -> str:
        proj_dir = f"--project  {os.path.abspath(options.get('proj_dir'))}/train/"
        if options.get("pretrained_weights_for_first_iteration") and current_state.current_task == "init":
            weights = f"--weights  {os.path.abspath(options.get('pretrained_weights_for_first_iteration'))}/"
        else:
            weights = f"--weights  {os.path.abspath(options.get('proj_dir'))}/train/weights/best.pt"
        train_batch_size = f"--batch-size {options.get('train_batch')}"
        train_epochs = f"--epochs {options.get('train_epochs')}"
        args = [proj_dir, weights, train_batch_size, train_epochs]
        return " ".join(args).join(YoloUltralytics.required_train_args)

    @staticmethod
    def call_test_script(options: dict, current_state: LoopState):
        owd = os.getcwd()
        os.chdir(options.get("detector_path"))
        if options.get("mute_stdout"):
            stdout_opt = subprocess.DEVNULL
        else:
            stdout_opt = subprocess.STDOUT
        args = YoloUltralytics.prepare_test_args(options, current_state)
        subprocess.call(["python3", "test.py", args], stdout=stdout_opt)
        os.chdir(owd)

    @staticmethod
    def call_train_script(options: dict, current_state: LoopState):
        owd = os.getcwd()
        os.chdir(options.get("detector_path"))
        if options.get("mute_stdout"):
            stdout_opt = subprocess.DEVNULL
        else:
            stdout_opt = subprocess.STDOUT
        args = YoloUltralytics.prepare_train_args(options, current_state)
        subprocess.call(["python3", "train.py", args], stdout=stdout_opt)
        os.chdir(owd)
