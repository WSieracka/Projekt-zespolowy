import yaml
import argparse


from loop_utils.loop_state import *
from yolo_ultralytics import YoloUltralytics
from Dataset import *


class DetectorHandle:
    @staticmethod
    def check_files(options: dict):
        if os.path.exists(options.get("detector_path")):
            return True
        else:
            return False

    @staticmethod
    def clone_repo(options: dict):
        pass

    @staticmethod
    def run_training(options: dict, dataset: Dataset, current_state: LoopState):
        if options.get("detector_name") == "ultralytics_yolo":
            YoloUltralytics.call_train_script(options, dataset, current_state)
        else:
            raise Exception

    @staticmethod
    def run_detection(options: dict, dataset: Dataset, current_state: LoopState):
        if options.get("detector_name") == "ultralytics_yolo":
            YoloUltralytics.call_test_script(options, dataset, current_state)
        else:
            raise Exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="detection", help='Task to execute (detection or training)')
    parser.add_argument("--cfg", type=str, required=True, help='Yaml file with options in active learning loop format')
    args = parser.parse_args()
    dataset = Dataset()
    with open(args.cfg) as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    state = LoopState(task=args.task, iteration=0)
    if args.task == "training":
        DetectorHandle.run_training(cfg, dataset, state)
    else:
        DetectorHandle.run_detection(cfg, dataset, state)
