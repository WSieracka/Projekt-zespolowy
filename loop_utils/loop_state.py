class LoopState:
    possible_tasks = ["detection", "training", "validation", "select_samples"]

    def __init__(self, task: str = "init", iteration: int = 0):
        self.current_task = task
        self.current_iteration = iteration