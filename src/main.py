from datasets.datasets import Datasets

dataset : Datasets = Datasets()

iterator = dataset.loadDataset("ET")
print(iterator.loadSamples("ETTh2", 4, ["HUFL", "HULL"]))