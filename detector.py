class Detector:

    def __init__(self, dir_path: str, proj_dir: str = None):
        self.dir_path = dir_path
        if proj_dir is not None:
            self.proj_dir = proj_dir

    def train(self):
        pass

    def test(self):
        pass
