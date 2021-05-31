import argparse


from loop_utils.loop_steps import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True,
                        help='"training" / "detection" / "validation" / "select_samples" / "finish_iteration"')
    parser.add_argument("--cfg", type=str, required=True,
                        help='Yaml config file')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    parser.parse_args()
    dataset = Dataset(os.path.abspath(cfg["whole_dataset_dir"]),
                      os.path.abspath(cfg["train_dataset_dir"]),
                      os.path.abspath(cfg["train_labels"]),
                      os.path.abspath(cfg["val_dataset_dir"]),
                      os.path.abspath(cfg["val_labels"]),
                      *cfg["dataset_classes_names"])
    current_state = LoopState(task=args.step, iteration=0)
    if LoopSteps.read_save_if_possible(cfg, dataset, current_state) is not None:
        dataset, current_state = LoopSteps.read_save_if_possible(cfg, dataset, current_state)
    if args.step == "training":
        if current_state.current_iteration == 0:
            LoopSteps.choose_initial_training_samples(cfg, dataset, current_state)
        else:
            LoopSteps.choose_new_training_samples(cfg, dataset, current_state)
        LoopSteps.save_state(cfg, dataset, current_state)
        LoopSteps.training(cfg, dataset, current_state)
    elif args.step == "detection":
        LoopSteps.detection(cfg, dataset, current_state)
        LoopSteps.save_state(cfg, dataset, current_state)
    elif args.step == "validation":
        LoopSteps.validation(cfg, dataset, current_state)
        LoopSteps.save_state(cfg, dataset, current_state)
    elif args.step == "select_samples":
        LoopSteps.select_samples(cfg, dataset, current_state)
    elif args.step == "finish_iteration":
        LoopSteps.finish_iteration(cfg, dataset, current_state)
        pass
    else:
        pass