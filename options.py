import yaml


class Options:

    def __init__(self, path: str):
        self.opt_dict = {}
        with open(path) as f:
            self.opt_dict = yaml.load(f, yaml.SafeLoader)

    def as_obj(self) -> object:
        class AttrOpt:
            pass
        attr_opt = AttrOpt()
        for key in self.opt_dict.keys():
            setattr(attr_opt, key, self.opt_dict[key])
        return attr_opt
