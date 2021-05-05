import yaml


f = open(file="chickens.yaml", mode="r")
y = yaml.load(stream=f, Loader=yaml.FullLoader)
print(*(y.values()))
