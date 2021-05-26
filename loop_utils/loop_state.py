class LoopState:
    possible_tasks = ["init", "detection", "training", "validation", "selection", "update_dataset", "single_call", "pull_new_samples"]

    def __init__(self, task: str = "init", iteration: int = 0):
        self.current_task = task
        self.current_iteration = iteration