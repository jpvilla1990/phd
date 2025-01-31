import time
import pandas as pd
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from uni2ts.eval_util.plot import plot_single

CONTEXT : int = 200
PREDICTION : int = 20

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLenght = CONTEXT,
)

iterator = dataset.loadDataset("power")
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration("power", True)
while True:
    sample : pd.core.frame.DataFrame = iterator.iterateDataset("power", ["Date_Time", "Natural Gas"])
    model.plotSample(
        sample.iloc[:CONTEXT],
        sample.iloc[CONTEXT:CONTEXT+PREDICTION],
    )
    if sample is None:
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