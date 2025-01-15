from datasets.datasets import Datasets

dataset : Datasets = Datasets()

#iterator = dataset.loadDataset("ET")
#iterator.getDatasetSizes()
#samples : list = iterator.loadSamples("ETTh2", 30, 3, ["HUFL", "HULL"])
#print(samples)

iterator = dataset.loadDataset("electricityUCI")
#iterator.getAvailableFeatures("electricity")
samples : list = iterator.loadSamples("electricity", 30, 1, ["MT_002", "MT_231"])
print(samples)
#print(iterator.getDatasetSizes())