import time
from datasets.datasets import Datasets

dataset : Datasets = Datasets()

iterator = dataset.loadDataset("power")
iterator.setSampleSize(16)
iterator.resetIteration("power", True)
while True:
    a = iterator.iterateDataset("power")
    print(a)
    if a is None:
        break
raise Exception("finished")
iterator.setSampleSize(16)
iterator.getAvailableFeatures("electricity")
startTime = time.perf_counter()
samples : list = iterator.loadSample("electricity",1 , 20, ["MT_002", "MT_231"])

iteration : int = 0
iterator.resetIteration("electricity", True)
while iteration < 10:
    iteration += 1
    print(iterator.iterateDataset(
        "electricity",
        ["MT_002", "MT_231"],
    ))

print(f"Elapsed time: {time.perf_counter() - startTime:.6f} seconds")