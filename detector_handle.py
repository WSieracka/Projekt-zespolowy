import os

from loop import LoopState
from yolo_ultralytics import YoloUltralytics


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
    def run_training(options: dict, current_state: LoopState):
        if options.get("detector_name") == "ultralytics_yolo":
            YoloUltralytics.call_train_script(options, current_state)
        else:
            raise Exception

    @staticmethod
    def run_detection(options: dict, current_state: LoopState):
        if options.get("detector_name") == "ultralytics_yolo":
            YoloUltralytics.call_test_script(options, current_state)
        else:
            raise Exception
