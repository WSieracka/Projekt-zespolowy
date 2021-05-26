import yaml
import argparse


from Dataset import Dataset
from loop_utils.loop_state import *
from loop_utils.loop_steps import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="detection",
                        help='Task to execute (detection or training)')
    parser.add_argument("--cfg", type=str, required=True,
                        help='Yaml config file')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    parser.parse_args()
    dataset = Dataset(cfg["whole_dataset_dir"],
                      cfg["train_dataset_dir"],
                      cfg["train_labels"],
                      cfg["val_dataset_dir"],
                      cfg["val_labels"],
                      *cfg["dataset_classes_names"])
    current_state = LoopState("init", 0)
    LoopSteps.choose_inital_training_samples(cfg, dataset, current_state)
    LoopSteps.save_state(cfg, dataset, current_state)
    # if args.step == "training":
    #     pass
    # elif args.step == "detection":
    #     pass
    # elif args.setp == "validation":
    #     pass
    # elif args.step == "update_dataset":
    #     pass
    # elif args.step == "select_samples":
    #     pass
    # else:
    #     pass