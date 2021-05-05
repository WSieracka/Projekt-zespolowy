class Detector:

    def __init__(self, dir_path: str, proj_dir: str = None, detector_opt_path: str = None):
        self.dir_path = dir_path
        if proj_dir is not None:
            self.proj_dir = proj_dir
        if detector_opt_path is not None:
            self.detector_opt_path = detector_opt_path

    def train(self):
        pass

    def test(self):
        pass
