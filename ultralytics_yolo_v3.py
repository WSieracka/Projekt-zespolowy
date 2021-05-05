import sys
import os

import yaml

sys.path.append("yolov3")

from detector import *
import yolov3.train as yolov3train
import yolov3.train as yolov3test
from yolov3.utils.torch_utils import select_device


class UltralyticsYolov3(Detector):
    def __init__(self, dir_path: str, proj_dir: str = None, detector_opt_path: str = None):
        super().__init__(dir_path, proj_dir, detector_opt_path)
        self.default_train_options = None
        self.default_test_options = None
        self.default_train_options()
        self.default_test_options()

    def train(self):
        self.set_default_train_options()
        opt = self.default_train_options
        if opt.device == "":
            device = select_device(opt.device, batch_size=opt.batch_size)
        else:
            device = opt.device
        with open(opt.hyp) as hyp_file:
            hyp = yaml.load(hyp_file, yaml.SafeLoader)
        yolov3train.train(hyp=hyp, opt=opt, device=device)

    def test(self):
        self.default_test_options()
        pass

    def set_default_train_options(self):
        opt_dict = {
            "weights": "yolov3.pt",
            "cfg": "",
            "data": "data/coco128.yaml",
            "hyp": "data/hyp.scratch.yaml",
            "epochs": 300,
            "batch_size": 16,
            "img_size": [640, 640],
            "rect": False,
            "resume": False,
            "nosave": False,
            "noautoanchor": False,
            "evolve": False,
            "bucket": "",
            "cache_images": False,
            "image_weights": False,
            "device": "",
            "multi_scale": False,
            "single_cls": False,
            "adam": False,
            "sync_bn": False,
            "local_rank": -1,
            "log_imgs": 16,
            "log_artifacts": False,
            "workers": 8,
            "project": "runs/train",
            "name": "exp",
            "exist-ok": False,
            "quad": False,

            "world_size": 1,
            "global_rank": -1,
            "save_dir": "test_save_dir/",
            "total_batch_size": 16
        }

        path_prefix_needed = ["weights", "data", "hyp"]
        for key in path_prefix_needed:
            opt_dict[key] = self.dir_path + opt_dict[key]

        class AttrOpt:
            pass
        attr_opt = AttrOpt()
        for key in opt_dict.keys():
            setattr(attr_opt, key, opt_dict[key])
        self.default_train_options = attr_opt

    def set_default_test_options(self):
        opt_dict = {
            "weights": None,
            "batch_size": 32,
            "imgsz": 640,
            "conf_thres": 0.001,
            "iou_thres": 0.6,
            "save_json": False,
            "single_cls": False,
            "augment": False,
            "verbose": False,
            "model": None,
            "dataloader": None,
            "save_dir": "",
            "save_txt": True,
            "save_hybrid": True,
            "save_conf": True,
            "plots": True,
            "log_imgs": 0
        }

        class AttrOpt:
            pass
        attr_opt = AttrOpt()
        for key in opt_dict.keys():
            setattr(attr_opt, key, opt_dict[key])
        self.default_test_options = attr_opt


if __name__ == "__main__":
    u = UltralyticsYolov3(dir_path="yolov3/")
    u.train()
