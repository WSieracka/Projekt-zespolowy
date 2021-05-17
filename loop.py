from detector_handle import DetectorHandle


class LoopState:
    def __init__(self, task: str = "init", iteration: int = 0):
        self.current_task = task
        self.current_iteration = iteration


class LoopSteps:
    @staticmethod
    def training():
        pass

    @staticmethod
    def validation():
        pass

    @staticmethod
    def detection():
        pass

    @staticmethod
    def check_stopping_condition(current_state: LoopState) -> bool:
        pass




class Loop:
    def __init__(self, cfg_file: str):
        self.current_state = LoopState()

    def run_step(self):
        pass

    def run_loop(self):
        pass


if __name__ == "__main__":
    pass